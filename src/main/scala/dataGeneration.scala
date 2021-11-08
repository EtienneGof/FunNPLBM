
import Common.DataGeneration.randomFunLBMDataGeneration
import Common.TSSInterface.{getSeriesPeriodogramsAndPcaCoefs, toTSS}
import Common.functionPrototypes._
import com.github.unsupervise.spark.tss.core.TSS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object dataGeneration {

  implicit val ss: SparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("AnalysePlan")
    .config("spark.executor.cores", 2)
    //.config("spark.executor.memory", "30G")
    .config("spark.executor.heartbeatInterval", "20s")
    .config("spark.driver.memory", "10G")
    .getOrCreate()

  ss.sparkContext.setLogLevel("WARN")
  ss.sparkContext.setCheckpointDir("checkpointDir")

  def main(args: Array[String]) {

    val sigma = 0.01
    val rowPartitionSize = List(20,30,40,30,20)
    val colPartitionSize = List(40,20,30,20,30)

    val prototypes: List[List[List[Double] => List[Double]]] =
      List(List(f1,f2,f3,f4,f5),
        List(f6,f7,f8,f2,f4),
        List(f9,f2,f4,f1,f5),
        List(f3,f8,f2,f4,f6),
        List(f5,f4,f9,f3,f1))

    val row = randomFunLBMDataGeneration(prototypes, sigma, rowPartitionSize, colPartitionSize, shuffle = false)
    val dataRDD: RDD[(Int, Int, List[Double], List[Double])] = ss.sparkContext.parallelize(row)
    val tss: TSS = toTSS(dataRDD)

    val (_, _, pcaCoefsByRow) = getSeriesPeriodogramsAndPcaCoefs(tss,
      10, 3, 0.9999)

    val mat = Common.ToolsRDD.RDDToMatrix(pcaCoefsByRow)

    require(mat.rows ==  rowPartitionSize.sum)
    require(mat.cols ==  colPartitionSize.sum)
    println(mat.rows, mat.cols)

    Common.IO.writeMatrixDvDoubleToCsv("src/main/scala/dataset.csv", mat)

  }
}
