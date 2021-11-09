package Common

import java.io.File

import Common.Tools._
import Common.ToolsRDD._
import breeze.linalg.{diag, DenseMatrix => BzDenseMatrix, DenseVector => BzDenseVector}
import breeze.stats.distributions.MultivariateGaussian
import com.github.unsupervise.spark.tss.core.TSS
import com.github.unsupervise.spark.tss.{functions => tssFunctions}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.{DenseMatrix, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SparkSession, functions}

object TSSInterface  {

  def getSeriesPeriodogramsAndPcaCoefs(tss: TSS, sizePeriodogram: Int = 10,
                                       nPcaAxisMax: Int = 5,
                                       thresholdVarExplained: Double = 0.99)(implicit ss: SparkSession) = {

    val periodogramAndSeries = getPeriodogramsAndSeries(tss, sizePeriodogram)
    val periodograms = periodogramAndSeries.map(row => (row._1, row._2, row._3))
    val series = periodogramAndSeries.map(row => (row._1, row._2, BzDenseVector(row._4.toArray)))
    val periodogramByRow = fromCellDistributionToRowDistribution(periodogramAndSeries.map(row => (row._1, row._2, BzDenseVector(row._3.toArray))))
    val seriesByRow = fromCellDistributionToRowDistribution(series)

    val p = periodogramByRow.take(1).head._2.length
    val dummyPartition = List.fill(p)(0)
    val partitionPerColBc: Broadcast[BzDenseVector[Int]] = ss.sparkContext.broadcast(BzDenseVector(dummyPartition: _*))

    val (meanList, covMat, _, _) = getMeansAndCovariances(periodogramByRow.map(r => (r._1, r._2, List(0))),
      colPartition = partitionPerColBc,
      KVec = List(1),
      fullCovariance = true,
      verbose = true)

    val (loadings,_) = getLoadingsAndVarianceExplained(covMat.flatten.head, nPcaAxisMax, thresholdVarExplained)
    val pcaCoefs = periodograms.map(r => (r._1, r._2,
      loadings * (BzDenseVector(r._3.toArray) - meanList.flatten.head)))

    val pcaCoefByRow = fromCellDistributionToRowDistribution(pcaCoefs)

    (seriesByRow, periodogramByRow, pcaCoefByRow)
  }

  def toTSS(data: RDD[(Int, Int, List[Double], List[Double])])(implicit ss: SparkSession): TSS = {

    val rddNewEncoding = data.map(row =>
      (row._1.toString,
        row._2.toString,
        row._3.head,
        row._3.last,
        row._3(1)- row._3.head,
        row._4))

    val dfWithSchema = ss.createDataFrame(rddNewEncoding)
      .toDF("scenario_id", "varName", "timeFrom", "timeTo","timeGranularity", "series")

    val dfWithDecorator = dfWithSchema.select(
      map(lit("scenario_id"),
        col("scenario_id"),
        lit("varName"),
        col("varName")).alias("decorators"),
      col("timeFrom"),
      col("timeTo"),
      col("timeGranularity"),
      col("series").alias("series")
    )

    TSS(dfWithDecorator)
  }

  def getPeriodogramsAndSeries(tss: TSS, sizePeriodogram:Int= 10)(implicit ss: SparkSession): RDD[(Int, Int, List[Double], List[Double])] = {

    val withFourierTable: TSS =
      tss.addZNormalized("zseries", TSS.SERIES_COLNAME, 0.0001)
        .addDFT("dft", "zseries")
        .addDFTFrequencies("dftFreq", TSS.SERIES_COLNAME, TSS.TIMEGRANULARITY_COLNAME)
        .addDFTPeriodogram("dftPeriodogram", "dft")

    val meanFourierFrequencyStep = withFourierTable
      .colSeqFirstStep("dftFreq")
      .agg(functions.mean("value"))
      .first.getDouble(0)

    val newInterpolationSamplePoints = (0 until sizePeriodogram).map(_.toDouble * meanFourierFrequencyStep)
    val minMaxAndMaxMinFourierFrequency = withFourierTable.series.select(min(array_max(col("dftFreq"))), max(array_min(col("dftFreq")))).first
    val minMaxFourierFrequency = minMaxAndMaxMinFourierFrequency.getDouble(0)
    val maxMinFourierFrequency = minMaxAndMaxMinFourierFrequency.getDouble(1)
    val keptInterpolationSamplePoints: Array[Double] = newInterpolationSamplePoints.filter(x => x < minMaxFourierFrequency && x > maxMinFourierFrequency).toArray

    val interpolatedFourierTSS = withFourierTable
      .addConstant("interpolatedDFTFreq", keptInterpolationSamplePoints)
      .addLinearInterpolationPoints("interpolatedDFTPeriodogram", "dftFreq", "dftPeriodogram", keptInterpolationSamplePoints)

    val logScaledTSS = interpolatedFourierTSS.addUDFColumn("logInterpolatedDFTPeriodogram",
      "interpolatedDFTPeriodogram",
      functions.udf(tssFunctions.log10(1D)
        .andThen((seq: Seq[Double]) => {Vectors.dense(seq.toArray)})))
      .repartition(200)
    val scaledTSS: TSS = logScaledTSS.addColScaled("logInterpolatedDFTPeriodogram_ScaledVecColScaled",
      "logInterpolatedDFTPeriodogram",scale = true,center = true)
    val seqScaledTSS = scaledTSS.addSeqFromMLVector("periodogram",
      "logInterpolatedDFTPeriodogram_ScaledVecColScaled")
    val series = seqScaledTSS.series

    val outputDf = series.select(
      seqScaledTSS.getDecoratorColumn("scenario_id").alias("scenario_id"),
      seqScaledTSS.getDecoratorColumn("varName").alias("varName"),
      col("periodogram"),
      col("series")).rdd

    outputDf.map(row => (
      row.getString(0).toInt,
      row.getString(1).toInt,
      row.getSeq[Double](2).toArray.toList,
      row.getSeq[Double](3).toArray.toList)
    )
  }

}
