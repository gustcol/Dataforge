/**
 * DataForge Scala Build Configuration
 *
 * SBT build file for DataForge Scala utilities.
 * Designed for Databricks Runtime 13.x / 14.x compatibility.
 */

name := "dataforge-scala"
version := "1.0.0"
scalaVersion := "2.12.15"

organization := "com.dataforge"
organizationName := "DataForge Team"

// Spark and Delta Lake versions (aligned with Databricks Runtime 14.x)
val sparkVersion = "3.5.0"
val deltaVersion = "3.0.0"

// Dependencies
libraryDependencies ++= Seq(
  // Spark Core (provided by Databricks)
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-streaming" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",

  // Delta Lake (provided by Databricks)
  "io.delta" %% "delta-spark" % deltaVersion % "provided",

  // Kafka integration
  "org.apache.spark" %% "spark-sql-kafka-0-10" % sparkVersion % "provided",

  // Testing
  "org.scalatest" %% "scalatest" % "3.2.17" % "test",
  "org.scalamock" %% "scalamock" % "5.2.0" % "test"
)

// Compiler options
scalacOptions ++= Seq(
  "-encoding", "UTF-8",
  "-deprecation",
  "-feature",
  "-unchecked",
  "-Xlint",
  "-Ywarn-dead-code",
  "-Ywarn-numeric-widen"
)

// Assembly settings for fat JAR
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case "reference.conf" => MergeStrategy.concat
  case x => MergeStrategy.first
}

// Exclude Spark from assembly (provided by cluster)
assembly / assemblyExcludedJars := {
  val cp = (assembly / fullClasspath).value
  cp filter { f =>
    f.data.getName.contains("spark") ||
    f.data.getName.contains("delta") ||
    f.data.getName.contains("hadoop")
  }
}

// Publishing settings
publishMavenStyle := true
publishTo := Some(
  "releases" at "https://your-maven-repo/releases"
)

// Documentation
Compile / doc / scalacOptions ++= Seq(
  "-doc-title", "DataForge Scala API",
  "-doc-version", version.value,
  "-groups"
)

// Test settings
Test / parallelExecution := false
Test / fork := true

// Console settings
console / initialCommands :=
  """
    |import org.apache.spark.sql.SparkSession
    |import org.apache.spark.sql.functions._
    |import com.dataforge._
    |
    |val spark = SparkSession.builder()
    |  .appName("DataForge Console")
    |  .master("local[*]")
    |  .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    |  .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    |  .getOrCreate()
    |
    |import spark.implicits._
    |println("DataForge console ready!")
  """.stripMargin
