package solutions.jagan.samples.sparkmllib

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

import java.time.format.DateTimeFormatter
import java.time.LocalDateTime
import scala.util.Random

object RandomForestCreditCardFraudClassifier {
  // this is the Kaggle Credit Card Fraud Detection dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  private val inputFilename = "../datasets/credit-card-fraud-detection/creditcard-small.csv"
  private val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH.mm.ss.SSS")
  private val noOfInputColumns = 30
  private val master = "local[64]"
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

  private def demonstrateReloadedModel(reloadedModel: RandomForestModel) = {

    val newData: Array[Double] = Array(34628.0) ++ Array.fill(noOfInputColumns - 2)(1.0) ++ Array(8.44)
    val newDataAsVector = Vectors.dense(newData)
    val prediction = reloadedModel.predict(newDataAsVector)

    println(s"\nReloaded model prediction on new data ${newDataAsVector} = " + prediction + ".")

  }

  private def reloadModel(model: RandomForestModel) = {

    val pathToModel = "model/CCFD_RF_" + LocalDateTime.now.format(formatter)
    model.save(sc, pathToModel)
    RandomForestModel.load(sc, pathToModel)

  }

  private def readData(fileName: String) = {

    // data obtained from the file with the header row skipped
    val data = sc.textFile(fileName).zipWithIndex.filter { case (_, index) => index != 0 }.map(_._1)

    val rows = data.collect.map {
      case line: String if line.split(",").length > 1 =>
        val splitLine = line.split(",")
        val inputElements = splitLine.slice(0, noOfInputColumns).map { elem => elem.toDouble }
        val fraudClass: Int = splitLine(noOfInputColumns).replace("\"", "").toInt
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
        LabeledPoint(row(noOfInputColumns), Vectors.dense(row.toArray.slice(0, noOfInputColumns)))
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

    // precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"threshold: $t, precision: $p")
    }

    // recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"threshold: $t, recall: $r")
    }

    // f-measure
    val fScore = metrics.fMeasureByThreshold
    fScore.foreach { case (t, f) =>
      println(s"threshold: $t, f-score: $f, beta = 1")
    }

    println
    println(s"area under precision-recall curve (AUPRC) = ${metrics.areaUnderPR}")
    println(s"area under receiver operating characteristic (AUROC) = ${metrics.areaUnderROC}")

  }

  private def performTraining(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint],
                              numClasses: Int): (RandomForestModel, BinaryClassificationMetrics) = {

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 100
    val featureSubsetStrategy = "auto"
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val model: RandomForestModel = RandomForest.trainClassifier(
      trainingData, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins
    )

    val predictionAndLabels = testData.map(p => (model.predict(p.features), p.label))
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    (model, metrics)
  }

  private def splitData(labeledData: RDD[LabeledPoint], seed: Long): (RDD[LabeledPoint], RDD[LabeledPoint]) = {

    val splits = labeledData.randomSplit(Array(0.7, 0.3), seed)
    val trainingData = splits(0)
    val testData = splits(1)

    (trainingData, testData)
  }

}
