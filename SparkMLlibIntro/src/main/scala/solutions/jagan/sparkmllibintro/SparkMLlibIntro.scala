package solutions.jagan.sparkmllibintro

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD

case class Flower(species: String)

// the following code is based on this article: https://www.baeldung.com/spark-mlib-machine-learning
object SparkMLlibIntro {
  val master = "local[2]"
  val objectName: String = this.getClass.getSimpleName.stripSuffix("$")
  val conf: SparkConf = new SparkConf().setAppName(objectName).setMaster(master)
  val sc: SparkContext = new SparkContext(conf)

  private def readData(fileName: String) = {

    val data = sc.textFile(fileName)

    println("Raw input data:\n")
    data.collect().foreach(println)
    println

    data.collect().map {
      case line: String if line.split(",").length > 1 => {
        val splitLine = line.split(",")
        val inputElements = List(splitLine(0), splitLine(1), splitLine(2), splitLine(3)).map { elem => elem.toDouble }
        val species = Flower(splitLine(4))
        val speciesAsNumber: Int = species match {
          case Flower("Iris-setosa") => 0
          case Flower("Iris-versicolor") => 1
          case Flower("Iris-virginica") => 2
          case _ => -1
        }
        Vectors.dense(inputElements.toArray :+ speciesAsNumber.toDouble)
      }
    }
  }

  private def printlPreprocessedInputData(rows: Array[Vector]): Unit = {

    println("Preprocessed input data:\n")
    rows.foreach { row =>
      print("specimen: ")
      row.toArray.foreach(elem => print(s"$elem, "))
      println
    }

  }

  def main(args: Array[String]): Unit = {

    println(s"\nJAG: $objectName\n")

    val inputFilename = "src/main/resources/iris/iris.data"
    val rows = readData(inputFilename)
    printlPreprocessedInputData(rows)

    val rowsAsRdd = sc.parallelize(rows)
    performExploratoryDataAnalysis(rowsAsRdd)


    println
  }

  private def performExploratoryDataAnalysis(data: RDD[Vector]): Unit = {

    val summary = Statistics.colStats(data)

    println("Summary mean:")
    println(summary.mean)
    println("Summary variance:")
    println(summary.variance)
    println("Summary non-zero:")
    println(summary.numNonzeros)

  }

}
