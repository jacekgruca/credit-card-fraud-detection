package solutions.jagan.sparkmllibintro

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

import java.time.format.DateTimeFormatter
import java.time.LocalDateTime
import scala.util.Random

object SparkMLlibIntro {
  // this is the Kaggle Credit Card Fraud Detection dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  private val inputFilename = "../datasets/credit-card-fraud-detection/creditcard-small.csv"
  private val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH.mm.ss.SSS")
  private val master = "local[100]"
  private val numClasses = 2

  private val objectName = this.getClass.getSimpleName.stripSuffix("$")
  private val conf = new SparkConf().setAppName(objectName).setMaster(master)
  private val sc = new SparkContext(conf)
  private val seed = new Random().nextLong()

  def main(args: Array[String]): Unit = {

    println(s"\nJAG: $objectName\n")

    val (rowsNoLabels, rowsWithLabels) = readData(inputFilename)
    val labeledData = getLabeledData(rowsWithLabels)
    printlLabeledData(labeledData)

    performExploratoryDataAnalysis(rowsNoLabels)
    displayCorrelationMatrix(rowsNoLabels)

    val (trainingData, testData) = splitData(labeledData, seed)
    val (model, metrics) = performTraining(trainingData, testData, numClasses)
    displayMetrics(metrics)
    val reloadedModel = reloadModel(model)
    demonstrateReloadedModel(reloadedModel)

    sc.stop()
  }

  private def demonstrateReloadedModel(reloadedModel: LogisticRegressionModel) = {

    val newData: Array[Double] = Array(1.0) ++ Array.fill(28)(1.0) ++ Array(1.0)
    val newDataAsVector = Vectors.dense(newData)
    val prediction = reloadedModel.predict(newDataAsVector)

    println(s"\nReloaded model prediction on new data ${newDataAsVector} = " + prediction + ".")

  }

  private def reloadModel(model: LogisticRegressionModel) = {

    val pathToModel = "model/CCFD_LR_" + LocalDateTime.now.format(formatter)
    model.save(sc, pathToModel)
    LogisticRegressionModel.load(sc, pathToModel)

  }

  private def readData(fileName: String) = {

    // data obtained from the file with the header row skipped
    val data = sc.textFile(fileName).zipWithIndex.filter { case (_, index) => index != 0 }.map(_._1)

    val rows = data.collect.map {
      case line: String if line.split(",").length > 1 =>
        val splitLine = line.split(",")
        val inputElements = splitLine.slice(0, 30).map { elem => elem.toDouble }
        val fraudClass: Int = splitLine(30).replace("\"", "").toInt
        (Vectors.dense(inputElements), Vectors.dense(inputElements :+ fraudClass.toDouble))
    }
    val rowsNoLabels = rows.map(_._1)
    val rowsWithLabels = rows.map(_._2)
    (rowsNoLabels, rowsWithLabels)
  }

  private def printlLabeledData(points: RDD[LabeledPoint]): Unit = {

    println("Top 50 rows of labeled input data:\n")
    points.take(50).foreach(println)
    println

    //wydrukować, ile jest pozytywnych, ile negatywnych, a potem prawdopodobieństwa (threshold) dla LR

    val fraudulentCount = points.filter(_.getLabel > 0.0).count()
    val totalCount = points.count()
    val legitimateCount = totalCount - fraudulentCount
    println("legitimateCount = " + legitimateCount)
    println("fraudulentCount = " + fraudulentCount)
    println("totalCount = " + totalCount)
    println

  }

  private def getLabeledData(rows: Array[Vector]): RDD[LabeledPoint] = {

    sc.parallelize {
      rows.map { row =>
        LabeledPoint(row(30), Vectors.dense(row.toArray.slice(0, 30)))
      }
    }

  }

  private def displayCorrelationMatrix(rowsWithoutLabel: Array[Vector]): Unit = {

    val correlMatrix = Statistics.corr(sc.parallelize(rowsWithoutLabel), "pearson")

    println
    println("Correlation matrix:")
    println(correlMatrix.toString)
    println

  }

  private def performExploratoryDataAnalysis(data: Array[Vector]): Unit = {

    val summary = Statistics.colStats(sc.parallelize(data))

    println("Summary mean:")
    println(summary.mean)
    println
    println("Summary variance:")
    println(summary.variance)
    println
    println("Summary non-zero:")
    println(summary.numNonzeros)
    println

  }

  private def displayMetrics(metrics: BinaryClassificationMetrics): Unit = {

    println

    // Precision by threshold
    val precision = metrics.pr
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

  }

  private def performTraining(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint],
                              numClasses: Int): (LogisticRegressionModel, BinaryClassificationMetrics) = {

    val model = new LogisticRegressionWithLBFGS().setNumClasses(numClasses).run(trainingData)
    val predictionAndLabels = testData.map(p => (model.predict(p.features), p.label))
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    (model, metrics)
  }

  private def splitData(labeledData: RDD[LabeledPoint], seed: Long): (RDD[LabeledPoint], RDD[LabeledPoint]) = {

    val splits = labeledData.randomSplit(Array(0.8, 0.2), seed)
    val trainingData = splits(0)
    val testData = splits(1)

    (trainingData, testData)
  }

}
