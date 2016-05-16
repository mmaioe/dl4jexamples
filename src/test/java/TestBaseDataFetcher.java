import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by ito_m on 5/13/16.
 */
public class TestBaseDataFetcher extends BaseDataFetcher {
    private double[][] data;
    public TestBaseDataFetcher(double[][] data){
        this.data = data;
        this.totalExamples = data.length;
        this.inputColumns = 2;
        this.numOutcomes = 2;
    }
    @Override
    public void fetch(int numExamples) {
        int from = cursor;
        int to = cursor + numExamples;
        if(to > totalExamples)
            to = totalExamples;

        List<DataSet> dataSet = new ArrayList<DataSet>();

        System.out.println("fetch:"+numExamples);
        for(int i=from;i<to;i++){
            dataSet.add(
                    new DataSet(
                            Nd4j.create(data[i]),
                            Nd4j.create(data[i])
                    )
            );
        }

        System.out.println("initialize : "+dataSet.size());
        initializeCurrFromList(dataSet);
        cursor += numExamples;
    }

    @Override
    protected void initializeCurrFromList(List<DataSet> examples) {

        if(examples.isEmpty())
            log.warn("Warning: empty dataset from the fetcher");
        curr = null;
        System.out.println("size?"+examples.size());
        INDArray inputs = createInputMatrix(examples.size());
        INDArray labels = createOutputMatrix(examples.size());
        for(int i = 0; i < examples.size(); i++) {
            INDArray data = examples.get(i).getFeatureMatrix();
            INDArray label = examples.get(i).getLabels();
            inputs.putRow(i, data);
            labels.putRow(i,label);
        }
        curr = new DataSet(inputs,labels);
        examples.clear();

    }
}
