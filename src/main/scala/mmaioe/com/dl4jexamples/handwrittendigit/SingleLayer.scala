package mmaioe.com.dl4jexamples.handwrittendigit

import mmaioe.com.dl4jdatareader.idx.IDXReader
import mmaioe.com.dl4jdatasetiterator.idx.IdxBaseDatasetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{OutputLayer, DenseLayer}
import org.deeplearning4j.nn.conf.{Updater, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
 * Created by ito_m on 5/18/16.
 */
object SingleLayer {
  def main(args: Array[String]) = {
    val numRows = 28
    val numColumns = 28
    val outputNum = 10
    val batchSize = 128
    val rngSeed = 123
    val numEpochs = 1

    val labelFile_test: String = "/Users/maoito/myProject/tensorflow/MNIST_data/t10k-labels-idx1-ubyte.gz"
    val attributeFIle_test: String = "/Users/maoito/myProject/tensorflow/MNIST_data/t10k-images-idx3-ubyte.gz"

    val labelFile_train: String = "/Users/maoito/myProject/tensorflow/MNIST_data/train-labels-idx1-ubyte.gz"
    val attributeFIle_train: String = "/Users/maoito/myProject/tensorflow/MNIST_data/train-images-idx3-ubyte.gz"

    //Get the DataSetIterators:
    val mnistTrain = IDXReader.read(attributeFIle_train,labelFile_train,1,1000)
    val mnistTest = IDXReader.read(attributeFIle_test,labelFile_test,1,1000)


    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.006)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .regularization(true).l2(1e-4)
      .list(2)
      .layer(0, new DenseLayer.Builder()
      .nIn(mnistTrain.inputColumns())
      .nOut(1000)
      .activation("relu")
      .weightInit(WeightInit.XAVIER)
      .build())
      .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
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
