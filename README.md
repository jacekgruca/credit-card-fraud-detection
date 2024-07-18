## Introduction

Fraud detection in financial transactions is a critical task for ensuring the integrity and security of financial systems. This note delves into leveraging the [Apache Spark](https://spark.apache.org/) framework to detect fraudulent activities using a dataset from Kaggle, which includes anonymized credit card transactions labeled as either fraudulent or genuine. By utilizing Spark's [MLlib](https://spark.apache.org/mllib/) and [Scala](https://scala-lang.org/) API, we demonstrate the process of data analysis, model training, and performance evaluation. This exploration highlights the capabilities of Spark in executing Machine Learning tasks efficiently.

## The dataset

Our dataset of choice is Kaggle's [Credit Card Fraud Detection collection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains anonymized credit card transactions labeled as fraudulent or genuine. The dataset contains 284 807 transactions and is some 151 MB, so it can be processed on a single machine. The set is pretty clean, however, some amounts are zero, which may or may not affect learning. 

## Code overview

### Configure build

Our Spark version used for this demonstration is 3.5.1 and our Scala version is 2.12.19, as shown in the following sbt configuration. You can view it and the remainder of the code [on GitHub](https://github.com/jacekgruca/credit-card-fraud-detection).

```scala
version := "0.1.0-SNAPSHOT"
scalaVersion := "2.12.19"

val sparkVersion = "3.5.1"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
)
```

In order to run the following example, [get sbt](https://www.scala-sbt.org/download), [get Spark](https://spark.apache.org/downloads.html) and execute:
```console
sbt package
/your/path/to/spark/bin/spark-submit --class solutions.jagan.samples.sparkmllib.RandomForestCreditCardFraudClassifier target/scala-2.12/creditcardfraudclassifier_2.12-0.1.0-SNAPSHOT.jar
```

The command might differ slightly depending on the operating system. This version was tested on macOS.

### Initialize Spark

Our project leverages Apache Spark. To get started, we first need to initialize it:

```scala
object RandomForestCreditCardFraudClassifier {
  private val objectName = this.getClass.getSimpleName.stripSuffix("$")
  private var sc: SparkContext = _
  private val master = "local[80]"

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName(objectName).master(master).getOrCreate()
    sc = spark.sparkContext

    // application logic
                                         
    spark.stop()
  }
```

The name of our object is RandomForestCreditCardFraudClassifier, because we will indeed use the [Random Forest method](https://en.wikipedia.org/wiki/Random_forest) for transaction classification (as fraudulent or not fraudulent). Just to clarify, the `objectName` value holds `"RandomForestCreditCardFraudClassifier"` and we used the builder pattern to create our `SparkSession`. The `SparkContext` is further used throughout the application, so we keep it as a `private var`. The `master val` specifies 80 tasks - a number which has been adjusted so that a single task on this dataset carries less than 1000 KiB of data (the default maximum recommended value). This configuration ensures that all tasks are executed locally on a single machine, but this is for demonstration purposes only. Spark is designed to run on a cluster of machines, and the `master` parameter can be used to control such distributed execution.

### Read data

Our further call is to read data, which is carried out by the following lines of code. All that the first line does is skip the header.

```scala
val data = sc.textFile(fileName).zipWithIndex.filter { case (_, index) => index != 0 }.map(_._1)

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
```

As you can see, we use regular expressions to validate the input. Parsing, however, is done by simple splitting on commas, as the format is pretty straightforward (once we know the data follows it).

Now that we have read in our data, it's time to create a `LabeledPoint` for each of the labeled transactions. This is needed because Spark's MLlib API utilizes the `LabeledPoint` class to train the model.

```scala
val points = rowsWithLabels.map { row =>
  LabeledPoint(row(noOfInputColumns), Vectors.dense(row.toArray.slice(0, noOfInputColumns)))
}
```

All `rowsNoLabels`, `rowsWithLabels`, and now `points` are instances of the RDD class. We decided to use RDD for this short intro, because RDDs ([Resilient Distributed Datasets](https://spark.apache.org/docs/latest/rdd-programming-guide.html#resilient-distributed-datasets-rdds)) are the foundation of Spark's distributed computing model and provide a simple way to understand the core concepts of working with data in Spark.

### Print a sample and a summary

Now that we have our data in the RDD format, we can print the top 50 rows and provide some statistics about the dataset. In the full collection we deal with 284 807 transactions, of which 492 were fraudulent. The following code prints all of this information.

```scala
println("Top 50 rows of labeled input data:\n")
points.take(50).foreach(println)

val fraudulentCount = points.filter(_.getLabel > 0.0).count()
val totalCount = points.count()
val legitimateCount = totalCount - fraudulentCount

println("legitimateCount = " + legitimateCount)
println("fraudulentCount = " + fraudulentCount)
println("totalCount = " + totalCount)
```

### Perform data analysis

As an exercise, we can perform some exploratory data analysis, just to see what Spark MLlib is made of.

```scala
val colStats = Statistics.colStats(rowsNoLabels)

println("stats mean: ")
println(colStats.mean)
println("stats variance: ")
println(colStats.variance)
println("stats non-zero: ")
println(colStats.numNonzeros)
```

### Print correlation matrix

The following code prints the correlation matrix, which is a great source of information for the Machine Learning Engineer. However, as this is a toolset exploration article, we will skip in-depth analysis of the matrix and just demonstrate that it is possible to obtain it with Spark MLlib.

```scala
val correlMatrix = Statistics.corr(rowsNoLabels, "pearson")

println("correlation matrix: ")
println(correlMatrix.toString)
```

### Split data

As discussed in various other articles, the dataset needs to be split into subsets, typically training, cross-validation and testing datasets. For this simple example, we'll skip cross-validation and focus on the training dataset, which will comprise 70% of the input data and the testing dataset, which will comprise the remainder of the input data. 

```scala
val splits = labeledData.randomSplit(Array(0.7, 0.3), seed)
val trainingData = splits(0)
val testData = splits(1)
```

### Train model

The time has come to perform model training. This is done with the following piece of code:

```scala
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
```

We're training the Random Forest classifier. At the beginning various training configuration is set, the details of which is beyond the scope of this tutorial. Next, we call the `trainClassifier` method and obtain a trained model. Lastly, we carry out the predictions on the test dataset and, based on them, we have our metrics created as BinaryClassificationMetrics (the Fraud Detection problem is, after all, a binary classification problem).

### Display metrics

Last but not least, we display our metrics. In Fraud Detection, certain performance metrics are particularly crucial to ensure the reliability of the model. Recall is vital because it measures the model's ability to identify actual fraudulent cases. High recall indicates that the model successfully detects most of the fraud instances, minimizing false negatives. Precision, on the other hand, assesses the accuracy of the positive predictions. High precision means that most of the transactions flagged as fraudulent are indeed fraudulent, reducing false positives. Recall and precision are computed based on then number of true positives, true negatives, false positives, and false negatives. The F-score, which combines precision and recall, provides a balanced measure of the model's accuracy. The most important item is the area under precision-recall curve (AUPRC) metric as recommended by the dataset authors. We also compute the area under receiver operating characteristic (AUROC), which is another balanced metric for Fraud Detection. For more details on these metrics see this [MLOps Blog post](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc).

```scala
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
```

### Persist and load the model

Lastly, the following lines of code persist the model to a location on the hard drive and then load it back. Afterwards, the reloaded model is called to make a prediction on some dummy data, just to demonstrate the reload cycle is possible and returns plausible results.

```scala
val pathToModel = "model/CCFD_RF_" + LocalDateTime.now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH.mm.ss.SSS"))
model.save(sc, pathToModel)
val reloadedModel = RandomForestModel.load(sc, pathToModel)

val newData: Array[Double] = Array(34628.0) ++ Array.fill(noOfInputColumns - 2)(1.0) ++ Array(8.44)
val newDataAsVector = Vectors.dense(newData)
val prediction = reloadedModel.predict(newDataAsVector)

println(s"\nreloaded model prediction on new data ${newDataAsVector} = " + prediction)

spark.stop()
```

This concludes our example, so we called the Spark stop() method.

## Sample output

The following is a sample Spark output generated by the above code (after removing boilerplate Spark logs). If you want to run it directly, you can [access it on GitHub](https://github.com/jacekgruca/credit-card-fraud-detection).

As you will see, we skipped 49 of the labeled input rows for brevity. The remaining parts of the output correspond to the sections above, so you can relate to them as needed.

```
RandomForestCreditCardFraudClassifier

Top 50 rows of labeled input data:

(0.0,[0.0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62])
(..)

legitimateCount = 284315
fraudulentCount = 492
totalCount = 284807

stats mean: 
[94813.85957508086,2.067790383364354E-15,1.192622389734055E-16,-1.7486012637846216E-15,1.9637069748057456E-15,1.3739009929736312E-15,1.214306433183765E-15,-1.1796119636642288E-16,2.5326962749261384E-16,-1.5543122344752192E-15,2.086872341600099E-15,1.5265566588595902E-15,-8.881784197001252E-16,8.847089727481716E-16,1.519617764955683E-15,4.579669976578771E-15,1.375635716449608E-15,-4.822531263215524E-16,4.926614671774132E-16,9.97465998686664E-16,7.806255641895632E-16,1.0928757898653885E-16,-1.2975731600306517E-15,2.0469737016526324E-16,4.513533644057155E-15,1.8735013540549517E-16,1.6679366221517E-15,-3.478120569333498E-16,-1.1752751549742868E-16,88.34961925093103]

stats variance: 
[2.2551240062021604E9,3.8364892520489025,2.726820024654341,2.2990292407266417,2.004683821505242,1.9050810468044528,1.7749462566038359,1.5304005706645336,1.4264788561143513,1.2069924674733965,1.1855938116171225,1.0418550849400998,0.9984034168389662,0.9905707931512168,0.9189055459213028,0.8378034011104347,0.7678191226564536,0.7213734477310357,0.7025393582201366,0.6626619368915566,0.5943253939972027,0.5395255276881478,0.5266427548263753,0.3899506607745836,0.36680837076763595,0.2717308268268892,0.23254289231868033,0.1629191909916957,0.10895496127868676,62560.06904632368]

stats non-zero: 
[284805.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,284807.0,282982.0]

correlation matrix: 
1.0                    0.11739630597838778      ... (30 total)
0.11739630597838778    1.0                      ...
-0.010593327121342908  -3.5543346031057056E-17  ...
-0.4196181722115278    -4.817146037752665E-16   ...
-0.10526020544572924   -3.109029677075789E-17   ...
0.1730721233103918     2.2679242601008653E-16   ...
-0.06301647037315038   1.4073084090930174E-16   ...
0.08471437480498976    1.0543176420500874E-15   ...
-0.03694943469000846   1.0783964961150547E-16   ...
-0.008660433697683663  -1.5990021405213728E-16  ...
0.030616628592320054   3.967911327412302E-17    ...
-0.24768943748667505   1.877799422751136E-16    ...
0.1243480683719315     -4.334110346433464E-18   ...
-0.06590202369761688   -7.371464877200687E-17   ...
-0.09875681920622528   5.051322062505067E-16    ...
-0.18345327348103374   6.590444555324477E-17    ...
0.011902867722433964   3.739819498242014E-16    ...
-0.07329721331779761   -3.0713121120338365E-16  ...
0.09043813254861077    1.1014300762165803E-16   ...
0.028975302561134585   2.0453616225006618E-16   ...
-0.05086601846835408   -5.498512930323036E-17   ...
0.04473572628908126    -2.49706915192897E-16    ...
0.14405905486146334    1.1008345949280314E-16   ...
0.051142364941772996   2.2975336880422676E-16   ...
-0.016181868459307656  -1.3661565660903186E-16  ...
-0.23308279059831197   2.8774144377287337E-16   ...
-0.04140710060605859   -1.0100443072120797E-16  ...
-0.005134591123997234  2.1205925296930928E-16   ...
-0.009412688179052262  -3.7044335107462927E-16  ...
-0.010596373389029466  -0.22770865292240752     ...

true positives: 122
true negatives: 85066
false positives: 17
false negatives: 40

precision: 0.8776978417266187
recall: 0.7530864197530864
F-score: 0.8106312292358804
area under precision-recall curve (AUPRC) = 0.769574701354915
area under receiver operating characteristic (AUROC) = 0.8764433074283456

reloaded model prediction on new data [34628.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,8.44] = 0.0
```

As shown, the area under precision-recall curve (AUPRC) for this execution is roughly 77%. That's pretty good by most standards! With further parameter tuning and a larger dataset, production grade metrics may be achieved.

## Conclusions and comparison with TensorFlow

As you can see, Spark provides a neat Scala API (Python API is also available) and functionality to perform Machine Learning on a cluster of machines with its MLlib. The library is geared towards general Machine Learning tasks on large datasets, however, for specialized ML tasks in deep learning other solutions might be faster. For example, [TensorFlow excels in deep learning workloads through its GPU acceleration](https://www.scaler.com/topics/tensorflow/gpus-for-deep-learning/), which is not natively supported by MLlib. Surprisingly, an up2date benchmark test of the two is yet to be delivered.

Ultimately, you can [combine Apache Spark and TensorFlow for even better performance](https://www.databricks.com/blog/2016/01/25/deep-learning-with-apache-spark-and-tensorflow.html). In fact, there are readily available solutions, which bring about the benefits of both, such as Yahoo's open-source [TensorFlowOnSpark](https://github.com/yahoo/TensorFlowOnSpark) and [SparkFlow](https://github.com/lifeomic/sparkflow). In addition, TensorFlow itself contains a package `spark-tensorflow-distributor`, which [helps you launch distributed training tasks using a Spark job](https://docs.databricks.com/en/_extras/notebooks/source/deep-learning/spark-tensorflow-distributor.html). 

It should not go unmentioned that Spark's latest Machine Learning package is called Spark ML, in contrast to MLlib. The difference between the two is discussed [on Spark's official website](https://spark.apache.org/docs/latest/ml-guide.html#announcement-dataframe-based-api-is-primary-api). Most importantly: MLlib utilizes Spark's [RDD](https://spark.apache.org/docs/latest/rdd-programming-guide.html), while Spark ML utilizes the [DataFrame](https://spark.apache.org/docs/latest/sql-programming-guide.html), which is high-level and similar in functionality to that known from [R](https://en.wikipedia.org/wiki/R_(programming_language)) or [Python pandas](https://en.wikipedia.org/wiki/Pandas_(software)). We decided to select MLlib as a topic for this article because we wanted to demonstrate the flexibility of Spark's RDDs and the control the library gives when training a model. However, it is not entirely impossible that further articles appear on this blog, this time involving Spark ML. Stay tuned!

## Further suggestions

As per the licensing, you are free to use this code and tune it to your needs. Performance-wise, the first thing to do would be profiling the solution to see which parts can be improved. A good starting point is [Spark's monitoring solutions](https://spark.apache.org/docs/latest/monitoring.html). A further step would involve obtaining a larger or a different dataset, depending on your interests. For example, Kumar Chandrakant in [his tutorial](https://www.baeldung.com/spark-mlib-machine-learning), which inspired this post, considers the "Hello, World" dataset of Machine Learning, the Iris Dataset. He performs a multiclass classification based on a number of floating point features - the dimensions of the flowers.

You can also think through, whether you want to carry two copies of the dataset across the Spark cluster. Currently, there is a Scala value `rowsNoLabels` and a Scala value `rowsWithLabels`, which carry much redundancy. They were both subsequently needed in the processing, so we decided to save time rather than space. You may make a different design decision.

Lastly, [a plethora of models](https://spark.apache.org/docs/latest/mllib-classification-regression.html) is available via Spark MLlib for Fraud Detection, including Logistic Regression, Decision Trees, Random Forest, Gradient-Boosted Trees, Support Vector Machines, and Naive Bayes. It's up to you to decide, which ones to try out.

In case of any questions or if you'd like to comment on this article, please get in touch with me.