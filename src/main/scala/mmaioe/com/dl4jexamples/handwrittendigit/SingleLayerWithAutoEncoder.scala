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
  def main(args: Array[String]) = {
    val outputNum = 10
    val numRows = 28
    val numColumns = 28
    val numOfExamples = 2000

    val labelFile_test: String = "/Users/maoito/myProject/tensorflow/MNIST_data/t10k-labels-idx1-ubyte.gz"
    val attributeFIle_test: String = "/Users/maoito/myProject/tensorflow/MNIST_data/t10k-images-idx3-ubyte.gz"

    val labelFile_train: String = "/Users/maoito/myProject/tensorflow/MNIST_data/train-labels-idx1-ubyte.gz"
    val attributeFIle_train: String = "/Users/maoito/myProject/tensorflow/MNIST_data/train-images-idx3-ubyte.gz"

    //1. Train Auto Encoder at first
    println(" start to train auto encoder...")

//    val mnistTrainAutoEncoder = IDXReader.readIdentical(attributeFIle_train,labelFile_train,500,numOfExamples,numRows*numColumns,outputNum)
    val mnistTrainAutoEncoder = IDXReader.readIdentical(attributeFIle_train,labelFile_train,1000,numRows*numColumns,outputNum)
    val featureExtractionOutputNum = 100
    val featureExtraction = new AutoEncoder(numRows * numColumns, featureExtractionOutputNum)

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
    val numEpochs = 1
    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.06)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .regularization(true).l2(1e-4)
      .list(2)
      .layer(0, new DenseLayer.Builder()
       .nIn(featureExtractionOutputNum)
       .nOut(1000)
       .activation("relu")
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(1, new OutputLayer.Builder(LossFunction.MCXENT)
        .nIn(1000)
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
    (0 until numEpochs).foreach(_  => model.fit(mnistTrain))

    println("Evaluate model....")
    val eval = new Evaluation(outputNum)
    while(mnistTest.hasNext()) {
      val next = mnistTest.next();
      val output = model.output(next.getFeatureMatrix());

      eval.eval(next.getLabels(), output);
    }


    println(eval.stats())
    println("****************Example finished********************")

  }
}
