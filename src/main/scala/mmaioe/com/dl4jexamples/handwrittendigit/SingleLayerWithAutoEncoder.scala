package mmaioe.com.dl4jexamples.handwrittendigit

import mmaioe.com.dl4jdatareader.idx.IDXReader
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{OutputLayer, DenseLayer}
import org.deeplearning4j.nn.conf.{Updater, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import mmaioe.com.featureextraction.representationlearning.AutoEncoder

/**
 * Created by ito_m on 6/4/16.
 */
object SingleLayerWithAutoEncoder {
  def test(layerOutputExamples: List[Int]): String={
    val outputNum = 10
    val numRows = 28
    val numColumns = 28
    val numOfExamples = 400
    val epoch = 2

    val labelFile_test: String = "/Users/maoito/myProject/tensorflow/MNIST_data/t10k-labels-idx1-ubyte.gz"
    val attributeFIle_test: String = "/Users/maoito/myProject/tensorflow/MNIST_data/t10k-images-idx3-ubyte.gz"

    val labelFile_train: String = "/Users/maoito/myProject/tensorflow/MNIST_data/train-labels-idx1-ubyte.gz"
    val attributeFIle_train: String = "/Users/maoito/myProject/tensorflow/MNIST_data/train-images-idx3-ubyte.gz"

    //1. Train Auto Encoder at first
    println(" start to train auto encoder...")

//        val mnistTrainAutoEncoder = IDXReader.readIdentical(attributeFIle_train,labelFile_train,500,numOfExamples,numRows*numColumns,numRows*numColumns,null)
    val mnistTrainAutoEncoder = IDXReader.readIdentical(attributeFIle_train,labelFile_train,1000,numRows*numColumns,numRows*numColumns,null)
    val featureExtractionOutputNum = 30
    val featureExtraction = new AutoEncoder(numRows * numColumns, featureExtractionOutputNum, layerOutputExamples,epoch)

    featureExtraction.training(mnistTrainAutoEncoder)
    //val featureExtractionOutputNum = numRows * numColumns
    //    val featureExtraction = null
    println(" finish to train auto encoder...")
    //    mnistTrainAutoEncoder.reset()

    //2. construct single layer neural net for features produced by autoencoder
    //    val mnistTrain = IDXReader.read(attributeFIle_train,labelFile_train,100,numOfExamples,featureExtractionOutputNum,outputNum,featureExtraction)
    //    val mnistTest = IDXReader.read(attributeFIle_test,labelFile_test,100,numOfExamples,featureExtractionOutputNum,outputNum,featureExtraction)

    val mnistTrain = IDXReader.read(attributeFIle_train,labelFile_train,100,featureExtractionOutputNum,outputNum,featureExtraction)
    val mnistTest = IDXReader.read(attributeFIle_test,labelFile_test,100,featureExtractionOutputNum,outputNum,featureExtraction)

    val rngSeed = 123
    val numEpochs = 10
    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.06)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .regularization(true).l2(1e-4)
      .list(4)
      .layer(0, new DenseLayer.Builder()
        .nIn(featureExtractionOutputNum)
        .nOut(300)
        .activation("sigmoid")
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(300)
        .nOut(150)
        .activation("sigmoid")
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(2, new DenseLayer.Builder()
        .nIn(150)
        .nOut(75)
        .activation("sigmoid")
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(3, new OutputLayer.Builder(LossFunction.MCXENT)
        .nIn(75)
        .nOut(outputNum)
        .activation("softmax")
        .weightInit(WeightInit.XAVIER)
        .build())
      .pretrain(false).backprop(true)
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    //    model.setListeners(new ScoreIterationListener(1))

    mnistTrain.reset();
    mnistTest.reset();

    println("Train model....")
    (0 until numEpochs).foreach {
      index =>
        model.fit(mnistTrain)
        mnistTrain.reset()
    }

    println("Evaluate model....")
    val eval = new Evaluation(outputNum)
    while(mnistTest.hasNext()) {
      val next = mnistTest.next();
      val output = model.output(next.getFeatureMatrix());

      eval.eval(next.getLabels(), output);
    }

    return "layer output:"+layerOutputExamples+"\n"+eval.stats()
  }

  def main(args: Array[String]) = {

    val stats = collection.mutable.ListBuffer[String]();

//    List(
//      List(500,250,125,62),
//      List(100,80,60,40),
//      List(600,300,150,70),
//      List(300,200,100,50)
//    ).foreach{
//      layer =>
//        stats += test(layer)
//    }

    List(
      List(1000),
      List(1200)
//      List(1200),
//      List(1400),
//      List(1600),
//      List(1800),
//      List(2000)
//      List(600), //=>1
//      List(500), //
//      List(400),
//      List(300),
//      List(200),
//      List(100),
//      List(50),
//      List(500,250), //=>2
//      List(100,50),
//      List(200,100),
//      List(700,350,150), //=>3
//      List(600,300,100),
//      List(500,250,100),
//      List(400,200,100),
//      List(300,200,100),
//      List(200,100,50)
    ).foreach{
      layer =>
        stats += test(layer)
    }

    println("Final Results ")
    stats.foreach{
      stat =>
        println(stat)

        println(">>>>>>>>>>>>>>>>>>>>>>>")
    }
  }
}
