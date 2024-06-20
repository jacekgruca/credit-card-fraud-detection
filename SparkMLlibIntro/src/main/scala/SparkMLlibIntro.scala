package solutions.jagan.spark.mllib.intro

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object SparkMLlibIntro {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("AnApp").setMaster("local")
    new SparkContext(conf)
    println()
    println("JAG: SparkMLlibIntro")
    println()
  }
}
