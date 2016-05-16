import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by ito_m on 5/13/16.
 */
public class TestJava {
    public static void main(String[] args) {
//    System.out.println(Nd4j.create(new double[]{1.0,2.2,3.9}));

        TestBaseDataFetcher fetcher = new TestBaseDataFetcher(
                new double[][]{
                        {1.0, 1.2},
                        {2.0, 1.1}
                }
        );

        BaseDatasetIterator iterator = new BaseDatasetIterator(
                1,
                2,
                fetcher
        );


        System.out.println("iterator Start:"+iterator.hasNext());
        while (iterator.hasNext()) {
            DataSet t = iterator.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();


            System.out.println(features);
        }
    }
}
