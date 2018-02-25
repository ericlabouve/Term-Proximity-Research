package DocumentClasses;

import java.util.Map;

/**
 * Method object for computing the Cosine Similarity between a
 * query and a document.
 *
 * @author ericlabouve
 */
public class CosineDistance implements DocumentDistance {

    /**
     * Computes the distance between the query and document
     *
     * cosine(d,q) = <d•q>/(||d|| * ||q||)
     * <d•q> = sum from i=1 to V: (Wij * Wiq)
     * ||x|| = sqrt(sum from i=1 to V: Wij^2)
     *
     * @param query The search query
     * @param document A document in documents
     * @param documents All documents in the database
     * @return The distance between the query and the document
     */
    @Override
    public double findDistance(TextVector query, TextVector document, VectorCollection documents) {
        double weightsProductRunningTotal = 0;
        // Loop through each word in |query| that maps to a word in |documents|
        // and sum the product of their weights
        for (Map.Entry<String, Double> pair: query.getNormalizedVectorEntrySet()) {
            // Document term weight * Query term weight
            weightsProductRunningTotal += document.getNormalizedFrequency(pair.getKey()) * pair.getValue();
        }

        double denominator  = document.getNormalizedL2Norm() * query.getNormalizedL2Norm();
        // Check for a divide by zero error
        if (denominator == 0) {
            return 0;
        }
        return weightsProductRunningTotal / denominator;
    }
}
