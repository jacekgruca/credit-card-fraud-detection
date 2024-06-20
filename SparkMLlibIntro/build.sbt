version := "0.1.0-SNAPSHOT"
scalaVersion := "2.12.19"

val sparkVersion = "3.5.1"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
)
