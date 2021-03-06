lazy val root = Project( id="dl4jexamples",  base=file(".")).
  settings(
    name := "dl4jexamples",
    version := "1.0",
    scalaVersion := "2.10.4",
    organization := "mmaioe.com",
    libraryDependencies ++= Seq(
      "commons-io" % "commons-io" % "2.4",
      "com.google.guava" % "guava" % "19.0",
      "jfree" % "jfreechart" % "1.0.13",
      "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc3.8",
      "org.deeplearning4j" % "deeplearning4j-nlp" % "0.4-rc3.8",
      "org.deeplearning4j" % "deeplearning4j-ui" % "0.4-rc3.8",
      "org.jblas" % "jblas" % "1.2.4",
      "org.nd4j" % "canova-nd4j-codec" % "0.0.0.14",
      "org.nd4j" % "nd4j-x86" % "0.4-rc3.8"
    )
  ).dependsOn(
    dl4jdatareader
  ).dependsOn(
    featureextraction
  )

lazy val dl4jdatareader = ProjectRef(file("../dl4jdatareader"), "dl4jdatareader")
lazy val featureextraction = ProjectRef(file("../featureextraction"), "featureextraction")
