package DocumentClasses;

import java.util.HashMap;
import java.util.Map;

/**
 * Method object for computing the Okapi BM25 distance between a
 * query and a document.
 *
 * @author ericlabouve
 */
public class OkapiDistance implements DocumentDistance {

    /** Keeps track of how many times a specific word appears in a collection of documents */
    private HashMap<String, Integer> wordDocumentCount;
    /** The average document length in a set of documents */
    private Double avdl;
    /** The total number of documents */
    private int N;
    /** The following values were given by the instructor */
    private static final double k1 = 1.2;
    private static final double k2 = 100;
    private static final double b = 0.75;

    public OkapiDistance(VectorCollection documents) {
        wordDocumentCount = new HashMap<String, Integer>();
        initWordDocumentCountMap(documents);
        avdl = getAvgDocLength(documents);
        N = documents.getSize();
    }

    /**
     * SUM( [ln( (N-dfi+0.5)/(dfi+0.5) )] * [ ((k1+1)*fij)/(k1*(1-b+b*dlj/avdl)+fij) ] * [((k2+1)*fiq)/(k2+fiq)])
     * ti is a term
     * fij is the raw frequency count of term ti in document dj
     * fiq is the raw frequency count of term ti in query q
     * N is the total number of documents in the collection
     * dfi is the number of documents that contain the term ti
     * dlj is the document length (in bytes) of d
     * avdl is the average document length of the collection
     * @param query RAW frequency for the query vector
     * @param document RAW frequency for the document vector
     * @param documents
     * @return
     */
    @Override
    public double findDistance(TextVector query, TextVector document, VectorCollection documents) {

        double sum = 0;
        // For each term in the query
        for (Map.Entry<String, Integer> pair : query.getRawVectorEntrySet()) {
            int dfi = getDocumentFrequency(pair.getKey());
            int fij = document.getRawFrequency(pair.getKey());
            int dlj = document.getTotalWordCount();
            int fiq = pair.getValue();

            /** Compute ln( (N-dfi+0.5)/(dfi+0.5) ) */
            double firstTerm = Math.log((N-dfi+0.5)/(dfi+0.5));
            /** Compute ((k1+1)*fij)/(k1*(1-b+b*dlj/avdl)+fij */
            double secondTerm = ((k1+1)*fij)/(k1*(1-b+b*dlj/avdl)+fij);
            /** Compute ((k2+1)*fiq)/(k2+fiq) */
            double thirdTerm = ((k2+1)*fiq)/(k2+fiq);

            double product = firstTerm * secondTerm * thirdTerm;
            sum += product;
        }
        return sum;
    }

    private double getAvgDocLength(VectorCollection documents) {
        double totalWordCount = 0;
        int count = 0;
        // Loop through each document
        for(int docNum = 1; docNum <= documents.getSize(); docNum++) {
            totalWordCount += documents.getVectorById(docNum).getTotalWordCount();
            count++;
        }
        return totalWordCount / count;
    }

    public void initWordDocumentCountMap(VectorCollection documents) {
        // For each Document in our collection
        for ( Map.Entry<Integer, TextVector> docPair: documents.getEntrySet()) {
            TextVector doc = docPair.getValue();
            // For each term in this document
            for (Map.Entry<String, Integer> wordFreqPair : doc.getRawVectorEntrySet()) {
                // If our word-document count map does not contain this word already
                if (! wordDocumentCount.containsKey(wordFreqPair.getKey())) {
                    // Add this word to our map
                    wordDocumentCount.put(wordFreqPair.getKey(), 0);
                }
                // Increment this word by one since it was discovered to be in this doc
                wordDocumentCount.put(wordFreqPair.getKey(), wordDocumentCount.get(wordFreqPair.getKey()) + 1);
            }
        }
    }

    public int getDocumentFrequency(String term) {
        Integer count = wordDocumentCount.get(term);
        if (count == null) {
            return 0;
        }
        return count;
    }
}
