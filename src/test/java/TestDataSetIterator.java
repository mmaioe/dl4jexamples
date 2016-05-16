import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

/**
 * Created by ito_m on 5/13/16.
 */
public class TestDataSetIterator extends BaseDatasetIterator {

    public TestDataSetIterator(int batch, int numExamples, BaseDataFetcher fetcher) {
        super(batch, numExamples, fetcher);
    }
}
