package labs;

import DocumentClasses.CosineDistance;
import DocumentClasses.VectorCollection;
import DocumentClasses.QueryVector;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Eric on 9/19/17.
 *
 * @author ericlabouve
 */
public class Lab2 {
    /** The Documents */
    public static VectorCollection documents;
    /** The Queries */
    public static VectorCollection queries;

    /**
     * Load data from your binary file.
     * Next, initialize the queries variable, and call the normalize() method on both variables.
     * Finally, print the 20 most relevant documents for each query.
     * @param args
     */
    public static void main(String[] args) {
        documents = new VectorCollection("/Users/Eric/Desktop/Thesis/programs/src/labs/documents.txt",
                VectorCollection.VectorType.DOCUMENTS,
                false, 2);
        queries = new VectorCollection("/Users/Eric/Desktop/Thesis/programs/src/labs/queries.txt",
                VectorCollection.VectorType.QUIRIES,
                false, 2);

        documents.normalize(documents);
        queries.normalize(documents);

        // Contains top 20 relevant documents in order for each query
        HashMap<Integer, ArrayList<Integer>> relevantDocuments = new HashMap<Integer, ArrayList<Integer>>();
        CosineDistance alg = new CosineDistance();

        for (int id = 1; id <= queries.getSize(); id++) {
            QueryVector qv = (QueryVector) queries.getVectorById(id);
            relevantDocuments.put(id, qv.findClosestDocuments(documents, alg));
            System.out.println(relevantDocuments.get(id));
        }
    }
}
