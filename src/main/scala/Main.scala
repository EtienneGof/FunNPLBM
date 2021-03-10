
import Common.IO
import Common.Tools._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Gamma

object Main {

        def main(args: Array[String]) {

                val shape = 1E1
                val scale = 2E1

                val alphaPrior = Some(Gamma(shape = shape, scale = scale)) // lignes
                val betaPrior = Some(Gamma(shape = shape, scale = scale)) // clusters redondants

                val trueRowPartitionSize = List(20,30,40,30,20)
                val trueColPartitionSize = List(40,20,30,20,30)
                val trueBlockPartition = getBlockPartition(getPartitionFromSize(trueRowPartitionSize),
                        getPartitionFromSize(trueColPartitionSize))

                val dataMatrix = Common.IO.readDenseMatrixDvDouble("src/main/scala/dataset.csv")
                val dataList = matrixToDataByCol(dataMatrix)

                val nIter = 5
                val verbose = false
                val nLaunches = 10


                println("Benchmark begins. It is composed of "+ nLaunches.toString + " launches, each launch runs every methods once.")

                (0 until nLaunches).foreach(iter => {

                        var t0 = System.nanoTime()

                        println("Launch number " + iter)

                        //////////////////////////////////// BGMM

                        val (ariBGMM, riBGMM, nmiBGMM, nClusterBGMM) = {

                                println("BGMM begins")

                                var t1 = System.nanoTime()

                                val bestBGMMRow = FunLBM.ModelSelection.gridSearch(
                                        dataMatrix,
                                        rangeRow = List(1,2,3,4,5,6,7),
                                        rangeCol = List(1),
                                        verbose  = verbose)
                                val bestBGMMRowPartition= bestBGMMRow("RowPartition").asInstanceOf[List[Int]]

                                val bestBGMMCol = FunLBM.ModelSelection.gridSearch(
                                        dataMatrix,
                                        rangeRow = List(1),
                                        rangeCol = List(1,2,3,4,5,6,7),
                                        verbose  = verbose)
                                val bestBGMMColPartition= bestBGMMCol("ColPartition").asInstanceOf[List[Int]]
                                val t = t1
                                t1 = printTime(t1, "BGMM")

                                val NPLBMPartition = getBlockPartition(bestBGMMRowPartition, bestBGMMColPartition)
                                getScores(NPLBMPartition, trueBlockPartition)
                        }

                        //////////////////////////////////// LBM

                        val (ariLBM, riLBM, nmiLBM, nClusterLBM) = {
                                println("FunLBM begins")

                                var t1 = System.nanoTime()
                                val bestLBM = FunLBM.ModelSelection.gridSearch(
                                        dataMatrix,
                                        rangeRow = List(1,2,3,4,5,6,7),
                                        rangeCol = List(1,2,3,4,5,6,7),
                                        verbose = verbose)
                                val (rowMembershipLBM, colMembershipLBM) = (
                                  bestLBM("RowPartition").asInstanceOf[List[Int]],
                                  bestLBM("ColPartition").asInstanceOf[List[Int]])
                                val NPLBMPartition = getBlockPartition(rowMembershipLBM, colMembershipLBM)
                                t1 = printTime(t1, "LBM")
                                getScores(NPLBMPartition, trueBlockPartition)
                        }
                        println(ariLBM, riLBM, nmiLBM, nClusterLBM)

                        ////////////////////////////////// BDPMM

                        val (ariBDPMM, riBDPMM, nmiBDPMM, nClusterBDPMM) = {
                                var t1 = System.nanoTime()
                                println("BDPMM begins")

                                val (rowMembershipsBDPMM, _, _) = new FunNPLBM.CollapsedGibbsSampler(
                                        dataList,
                                        alphaPrior = alphaPrior,
                                        betaPrior  = betaPrior,
                                        initByUserColPartition = Some(dataList.indices.toList)).runWithFixedPartitions(
                                        nIter,
                                        verbose = verbose)

                                val (_ ,colMembershipsBDPMM , _) = new FunNPLBM.CollapsedGibbsSampler(
                                        dataList,
                                        alphaPrior = alphaPrior,
                                        betaPrior  = betaPrior,
                                        initByUserRowPartition = Some(dataList.head.indices.toList)
                                ).runWithFixedPartitions(
                                        nIter,
                                        verbose = verbose, updateCol = true, updateRow = false)

                                val BDPMMPartition = getBlockPartition(rowMembershipsBDPMM.last, colMembershipsBDPMM.last)
                                t1 = printTime(t1, "BDPMM")
                                getScores(BDPMMPartition, trueBlockPartition)
                        }

                        //////////////////////////////////// FunNPLBM

                        val (ariNPLBM, riNPLBM, nmiNPLBM, nClusterNPLBM) = {
                                println("FunNPLBM begins")

                                var t1 = System.nanoTime()
                                val (rowMembershipNPLBM, colMembershipNPLBM, _) = new FunNPLBM.CollapsedGibbsSampler(dataList,
                                        alphaPrior = alphaPrior,
                                        betaPrior = betaPrior).run(nIter, verbose = verbose)
                                t1 = printTime(t1, "FunNPLBM")
                                val NPLBMPartition = getBlockPartition(rowMembershipNPLBM.last, colMembershipNPLBM.last)
                                getScores(NPLBMPartition, trueBlockPartition)
                        }

                        val ARIs = Array(shape, scale, ariBGMM, ariLBM, ariBDPMM, ariNPLBM)
                        val RIs  = Array(shape, scale, riBGMM, riLBM, riBDPMM, riNPLBM)
                        val NMIs = Array(shape, scale, nmiBGMM, nmiLBM, nmiBDPMM, nmiNPLBM)
                        val nClusters = Array(shape, scale, nClusterBGMM, nClusterLBM, nClusterBDPMM, nClusterNPLBM)

                        val ARIMat = DenseMatrix(ARIs.map(_.toString)).reshape(1, ARIs.length)
                        val RIMat  = DenseMatrix( RIs.map(_.toString)).reshape(1, ARIs.length)
                        val NMIMat = DenseMatrix(NMIs.map(_.toString)).reshape(1, ARIs.length)
                        val nClusterMat = DenseMatrix(nClusters.map(_.toString)).reshape(1, ARIs.length)

                        val append = true

                        println(ARIMat)
                        println(RIMat)
                        println(NMIMat)
                        println(nClusterMat)

                        //        IO.writeMatrixStringToCsv("src/main/scala/Benchmark/ARIs.csv", ARIMat, append=append)
                        //        IO.writeMatrixStringToCsv("src/main/scala/Benchmark/RIs.csv" , RIMat , append=append)
                        //        IO.writeMatrixStringToCsv("src/main/scala/Benchmark/NMIs.csv", NMIMat, append=append)
                        //        IO.writeMatrixStringToCsv("src/main/scala/Benchmark/nClusters.csv", nClusterMat, append=append)

                })
        }
}

