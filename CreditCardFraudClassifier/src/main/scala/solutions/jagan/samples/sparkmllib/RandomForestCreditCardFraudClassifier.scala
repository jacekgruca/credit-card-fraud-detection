package solutions.jagan.samples.sparkmllib

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import java.time.format.DateTimeFormatter
import scala.util.matching.Regex
import java.time.LocalDateTime
import scala.util.Random

/**
 * This object contains all logic necessary to demonstrate the detection of fraudulent credit card transactions.
 * It utilizes the Random Forest classification method available in the Apache Spark framework's MLlib library.
 */
object RandomForestCreditCardFraudClassifier {
  // this is the Kaggle Credit Card Fraud Detection dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  private val inputFilename = "../datasets/credit-card-fraud-detection/creditcard-full.csv"
  private val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH.mm.ss.SSS")
  private val seed = new Random().nextLong()
  private val noOfInputColumns = 30
  // this master value has been adjusted so that a single task on this dataset is less or equal to 1000 KiB
  private val master = "local[80]"
  // this is a binary classification problem (each transaction is either fraudulent or not),
  // so the number of classes is 2
  private val numClasses = 2

  // objectName holds "RandomForestCreditCardFraudClassifier"
  private val objectName = this.getClass.getSimpleName.stripSuffix("$")
  private var sc: SparkContext = _

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName(objectName).master(master).getOrCreate()
    sc = spark.sparkContext

    println(s"\n$objectName\n")

    val (rowsNoLabels, rowsWithLabels) = readData(inputFilename)
    val labeledData = getLabeledData(rowsWithLabels)
    printlLabeledData(labeledData)

    performExploratoryDataAnalysis(rowsNoLabels)
    displayCorrelationMatrix(rowsNoLabels)

    val (trainingData, testData) = splitData(labeledData, seed)
    val (model, predictions, metrics) = performTraining(trainingData, testData, numClasses)
    displayMetrics(predictions, metrics)

    val reloadedModel = reloadModel(model)
    demonstrateReloadedModel(reloadedModel)

    spark.stop()
  }

  private def demonstrateReloadedModel(reloadedModel: RandomForestModel): Unit = {

    val newData: Array[Double] = Array(34628.0) ++ Array.fill(noOfInputColumns - 2)(1.0) ++ Array(8.44)
    val newDataAsVector = Vectors.dense(newData)
    val prediction = reloadedModel.predict(newDataAsVector)

    println(s"\nreloaded model prediction on new data $newDataAsVector = " + prediction)

  }

  // this method persists the model to a location on the hard drive and loads the persisted model back
  private def reloadModel(model: RandomForestModel) = {

    val pathToModel = "model/CCFD_RF_" + LocalDateTime.now.format(formatter)
    model.save(sc, pathToModel)
    RandomForestModel.load(sc, pathToModel)

  }

  private def readData(fileName: String) = {

    // data obtained from the file with the header row skipped
    val data = sc.textFile(fileName).zipWithIndex.filter { case (_, index) => index != 0 }.map(_._1)

    // this logic processes all dataset lines and places the input columns of each in the first element of the tuple and
    // the input columns plus label (fraudClass) in the second element on the tuple; this way we can operate on both
    // labeled and unlabeled data
    val integerRegex = """-?\d+([eE][-+]?\d+)?"""
    val doubleRegex = """-?\d+(\.\d+)?([eE][-+]?\d+)?"""
    val classRegex = s"""\"$doubleRegex\""""
    val pattern: Regex = new Regex(integerRegex + "," + (doubleRegex + ",") * (noOfInputColumns - 1) + classRegex)
    val rowsWithLabels = data.map {
      line =>
        line.replaceAll("\\s", "") match {
          case pattern(_*) =>
            val splitLine = line.split(",")
            val inputElements = splitLine.slice(0, noOfInputColumns).map(elem => elem.toDouble)
            val fraudClass: Int = splitLine(noOfInputColumns).replace("\"", "").toInt
            Vectors.dense(inputElements :+ fraudClass.toDouble)
          case l => throw new IllegalArgumentException(s"$line is not valid input.")
        }
    }
    val rowsNoLabels = rowsWithLabels.map(row => Vectors.dense(row.toArray.dropRight(1)))

    (rowsNoLabels, rowsWithLabels)
  }

  private def printlLabeledData(points: RDD[LabeledPoint]): Unit = {

    println("Top 50 rows of labeled input data:\n")
    points.take(50).foreach(println)
    println

    val fraudulentCount = points.filter(_.getLabel > 0.0).count()
    val totalCount = points.count()
    val legitimateCount = totalCount - fraudulentCount

    println("legitimateCount = " + legitimateCount)
    println("fraudulentCount = " + fraudulentCount)
    println("totalCount = " + totalCount)
    println

  }

  private def getLabeledData(rowsWithLabels: RDD[Vector]): RDD[LabeledPoint] = {

    rowsWithLabels.map { row =>
      LabeledPoint(row(noOfInputColumns), Vectors.dense(row.toArray.slice(0, noOfInputColumns)))
    }

  }

  private def displayCorrelationMatrix(rowsNoLabels: RDD[Vector]): Unit = {

    val correlMatrix = Statistics.corr(rowsNoLabels, "pearson")

    println
    println("correlation matrix: ")
    println(correlMatrix.toString)
    println

  }

  private def performExploratoryDataAnalysis(rowsNoLabels: RDD[Vector]): Unit = {

    val colStats = Statistics.colStats(rowsNoLabels)

    println("stats mean: ")
    println(colStats.mean)
    println
    println("stats variance: ")
    println(colStats.variance)
    println
    println("stats non-zero: ")
    println(colStats.numNonzeros)
    println

  }

  private def displayMetrics(predictions: RDD[(Double, Double)], metrics: BinaryClassificationMetrics): Unit = {

    val truePositives = predictions.filter {
      case (prediction, label) => prediction == 1.0 && label == 1.0
    }.count()

    val trueNegatives = predictions.filter {
      case (prediction, label) => prediction == 0.0 && label == 0.0
    }.count()

    val falsePositives = predictions.filter {
      case (prediction, label) => prediction == 1.0 && label == 0.0
    }.count()

    val falseNegatives = predictions.filter {
      case (prediction, label) => prediction == 0.0 && label == 1.0
    }.count()

    println(s"true positives: $truePositives")
    println(s"true negatives: $trueNegatives")
    println(s"false positives: $falsePositives")
    println(s"false negatives: $falseNegatives")

    val precision = truePositives.toDouble / (truePositives + falsePositives)
    val recall = truePositives.toDouble / (truePositives + falseNegatives)

    println
    println("precision: " + precision)
    println("recall: " + recall)
    val fScore = 2 * precision * recall / (precision + recall)
    println("F-score: " + fScore)
    println(s"area under precision-recall curve (AUPRC) = ${metrics.areaUnderPR}")
    println(s"area under receiver operating characteristic (AUROC) = ${metrics.areaUnderROC}")

  }

  private def performTraining(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint], numClasses: Int):
  (RandomForestModel, RDD[(Double, Double)], BinaryClassificationMetrics) = {

    // the following are the parameters of the training model
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 100
    val featureSubsetStrategy = "auto"
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val model: RandomForestModel = RandomForest.trainClassifier(
      trainingData, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins
    )

    val predictions = testData.map(p => (model.predict(p.features), p.label))
    val metrics = new BinaryClassificationMetrics(predictions)

    (model, predictions, metrics)
  }

  private def splitData(labeledData: RDD[LabeledPoint], seed: Long): (RDD[LabeledPoint], RDD[LabeledPoint]) = {

    // our split of data is 70% to 30%
    val splits = labeledData.randomSplit(Array(0.7, 0.3), seed)
    val trainingData = splits(0)
    val testData = splits(1)

    (trainingData, testData)
  }

}
