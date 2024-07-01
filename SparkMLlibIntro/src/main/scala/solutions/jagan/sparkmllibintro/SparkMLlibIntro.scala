package solutions.jagan.sparkmllibintro

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

import java.time.format.DateTimeFormatter
import java.time.LocalDateTime
import scala.util.Random

private case class Flower(species: String)

// the following code is based on this article: https://www.baeldung.com/spark-mlib-machine-learning
object SparkMLlibIntro {
  private val master = "local"
  private val objectName = this.getClass.getSimpleName.stripSuffix("$")
  private val conf = new SparkConf().setAppName(objectName).setMaster(master)
  private val sc = new SparkContext(conf)
  private val inputFilename = "src/main/resources/iris/iris.data"
  private val seed = new Random().nextLong()
  private val noOfClasses = 3
  private val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH.mm.ss.SSS")

  def main(args: Array[String]): Unit = {

    println(s"\nJAG: $objectName\n")

    val (rowsNoLabels, rowsWithLabels) = readData(inputFilename)
    val labeledData = getLabeledData(rowsWithLabels)

    printlLabeledData(labeledData)
    performExploratoryDataAnalysis(rowsNoLabels)
    displayCorrelationMatrix(rowsNoLabels)

    val (trainingData, testData) = splitData(labeledData, seed)
    val (model, metrics) = performTraining(trainingData, testData, noOfClasses)
    displayMetrics(metrics)
    val reloadedModel = reloadModel(model)
    val newData = Vectors.dense(Array[Double](1, 1, 1, 1))
    val prediction = reloadedModel.predict(newData)
    println(s"Reloaded model prediction on new data ${newData} = " + getSpeciesAsName(prediction).species + ".")

    println
    sc.stop()
  }

  private def getSpeciesAsName(prediction: Double) = {
    prediction.toInt match {
      case 0 => Flower("Iris-setosa")
      case 1 => Flower("Iris-versicolor")
      case 2 => Flower("Iris-virginica")
      case other => throw new MatchError(s"Prediction ($other) didn't match any of the known iris species.")
    }
  }

  private def getSpeciesAsNumber(species: Flower) = {
    species match {
      case Flower("Iris-setosa") => 0
      case Flower("Iris-versicolor") => 1
      case Flower("Iris-virginica") => 2
      case other => throw new MatchError(s"Iris species name (${other.species}) unknown.")
    }
  }

  private def reloadModel(model: LogisticRegressionModel) = {

    val pathToModel = "model/LR_" + LocalDateTime.now.format(formatter)
    model.save(sc, pathToModel)
    LogisticRegressionModel.load(sc, pathToModel)

  }

  private def readData(fileName: String) = {

    val data = sc.textFile(fileName)

    println("Raw input data:\n")
    data.collect().foreach(println)
    println

    val rows = data.collect().map {
      case line: String if line.split(",").length > 1 =>
        val splitLine = line.split(",")
        val inputElements = List(splitLine(0), splitLine(1), splitLine(2), splitLine(3)).map { elem => elem.toDouble }
        val species = Flower(splitLine(4))
        val speciesAsNumber: Int = getSpeciesAsNumber(species)
        (Vectors.dense(inputElements.toArray), Vectors.dense(inputElements.toArray :+ speciesAsNumber.toDouble))
    }
    val rowsNoLabels = rows.map(_._1)
    val rowsWithLabels = rows.map(_._2)
    (rowsNoLabels, rowsWithLabels)
  }

  private def printlLabeledData(points: RDD[LabeledPoint]): Unit = {

    println("Labeled input data:\n")
    points.collect().foreach(println)

  }

  private def getLabeledData(rows: Array[Vector]): RDD[LabeledPoint] = {

    sc.parallelize {
      rows.map { row =>
        LabeledPoint(row(4), Vectors.dense(row.toArray.slice(0, 4)))
      }
    }

  }

  private def displayCorrelationMatrix(rowsWithoutLabel: Array[Vector]): Unit = {

    val correlMatrix = Statistics.corr(sc.parallelize(rowsWithoutLabel), "pearson")

    println("Correlation matrix:")
    println(correlMatrix.toString)

  }

  private def performExploratoryDataAnalysis(data: Array[Vector]): Unit = {

    val summary = Statistics.colStats(sc.parallelize(data))

    println("Summary mean:")
    println(summary.mean)
    println("Summary variance:")
    println(summary.variance)
    println("Summary non-zero:")
    println(summary.numNonzeros)

  }

  private def displayMetrics(metrics: MulticlassMetrics): Unit = {

    println("Model accuracy on test data: " + metrics.accuracy)
    println(s"Confusion matrix:\n${metrics.confusionMatrix}")
    println(s"Precision: ${metrics.weightedPrecision}")
    println(s"Recall: ${metrics.weightedRecall}")
    println(s"F1 Score: ${metrics.weightedFMeasure}")

  }

  private def performTraining(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint],
                              numClasses: Int): (LogisticRegressionModel, MulticlassMetrics) = {

    val model = new LogisticRegressionWithLBFGS().setNumClasses(numClasses).run(trainingData);
    val predictionAndLabels = testData.map(p => (model.predict(p.features), p.label))
    val metrics = new MulticlassMetrics(predictionAndLabels)

    (model, metrics)
  }

  private def splitData(labeledData: RDD[LabeledPoint], seed: Long): (RDD[LabeledPoint], RDD[LabeledPoint]) = {

    val splits = labeledData.randomSplit(Array(0.8, 0.2), seed)
    val trainingData = splits(0)
    val testData = splits(1)

    (trainingData, testData)
  }

}
