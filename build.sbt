lazy val root = Project( id="dl4jexamples",  base=file(".")).
  settings(
    name := "dl4jexamples",
    version := "1.0",
    scalaVersion := "2.10.4",
    organization := "mmaioe.com",
    libraryDependencies ++= Seq(
      "org.deeplearning4j" % "deeplearning4j-nlp" % "0.4-rc3.8",
      "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc3.8",
      "org.deeplearning4j" % "deeplearning4j-ui" % "0.4-rc3.8",
      "com.google.guava" % "guava" % "19.0",
      "org.nd4j" % "canova-nd4j-image" % "0.0.0.14",
      "org.nd4j" % "canova-nd4j-codec" % "0.0.0.14"
    )
  )
