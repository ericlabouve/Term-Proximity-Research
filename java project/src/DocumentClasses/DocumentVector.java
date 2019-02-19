package DocumentClasses;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Represents a single document in the vector collection.
 * normalizedTermToFreqMap will be filled out once normalize(VectorCollection)
 * is invoked.
 *
 * @author ericlabouve
 */
public class DocumentVector extends TextVector {
    private HashMap<String, Double> normalizedTermToFreqMap = new HashMap<String, Double>();

    @Override
    public Set<Map.Entry<String, Double>> getNormalizedVectorEntrySet() {
        return normalizedTermToFreqMap.entrySet();
    }

    /**
     * Normalizes this class's termToFreqMap according to the database
     * represented by vc and stores the results in normalizedTermToFreqMap.
     *
     * W = tf * idf
     * tf = f / max{f1, f2, ..., f|V|} where f is the raw frequency count of a term t
     * idf = log2(N/df) where N is the total number of documents in the system and df
     *      is the number of documents in which the term t appears at least once.
     *
     * @param vc The database of vectors to normalize termToFreqMap against.
     */
    @Override
    public void normalize(VectorCollection vc) {
        // Obtain the highest tf in the document
        double maxF = (double) getHighestRawFrequency();
        // Loop through all the (term, frequency) pairs in our Document
        for (Map.Entry<String, Integer> pair : getRawVectorEntrySet()) {
            double tf = pair.getValue() / maxF;
            double df = vc.getDocumentFrequency(pair.getKey());
            double idf;
            if (df == 0) {
                idf = 0;
            } else {
                idf = log2(vc.getSize() / df);
            }
            double w = tf * idf;
            // Populate the document vector with the new normalized frequency
            normalizedTermToFreqMap.put(pair.getKey(), w);
        }
    }

    @Override
    public double getNormalizedFrequency(String word) {
        return normalizedTermToFreqMap.get(word) != null ? normalizedTermToFreqMap.get(word) : 0;
    }
}
