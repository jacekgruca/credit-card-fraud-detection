package solutions.jagan.spark.mllib.intro

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

case class Flower(species: String)

// the following code is based on this article: https://www.baeldung.com/spark-mlib-machine-learning
object SparkMLlibIntro {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("SparkMLlibIntro").setMaster("local[2]")
    val sc = new SparkContext(conf)

    println()
    println("JAG: SparkMLlibIntro")
    println()

    val inputFile = "src/main/resources/iris/iris.data"
    val data = sc.textFile(inputFile)

    println("Raw input data:\n")
    data.collect().foreach(println)
    println

    val rows = data.collect().map {
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
        (inputElements, speciesAsNumber)
      }
    }

    println("Preprocessed input data:\n")
    rows.foreach { row =>
      print("specimen: ")
      row._1.foreach(elem => print(s"$elem, "))
      print(s"${row._2}")
      println
    }

    println
  }
}
