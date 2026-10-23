ThisBuild / scalaVersion := "2.12.18"
ThisBuild / organization := "org.example"
ThisBuild / version := "0.1.0"

lazy val root = (project in file("."))
  .settings(
    name := "movie-graphx-analysis",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "3.5.1" % "provided",
      "org.apache.spark" %% "spark-graphx" % "3.5.1" % "provided"
    )
  )
