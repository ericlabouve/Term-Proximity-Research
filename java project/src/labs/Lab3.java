package labs;

import DocumentClasses.*;
import DocumentClasses.VectorCollection;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * @author ericlabouve
 */
public class Lab3 {
    /** The Documents */
    public static VectorCollection documents;
    /** The Queries */
    public static VectorCollection queries;
    /** Contains top 20 relevant documents in order for each query according to Consine Similarity */
    public static HashMap<Integer, ArrayList<Integer>> cosineDocumentResults;
    /** Contains top 20 relevant documents in order for each query according to Okapi Distance */
    public static HashMap<Integer, ArrayList<Integer>> okapiDocumentResults;
    /** The variable contains the IDs of the relevant documents for each query */
    public static HashMap<Integer, ArrayList<Integer>> humanJudgement;

    /**
     * Compares MAP scores between cosine and Okapi BM25
     * @param args
     */
    public static void main(String[] args) {
        documents = new VectorCollection("/Users/Eric/Desktop/Thesis/programs/src/labs/documents.txt",
                VectorCollection.VectorType.DOCUMENTS,
                false, 2);
        queries = new VectorCollection("/Users/Eric/Desktop/Thesis/programs/src/labs/queries.txt",
                VectorCollection.VectorType.QUIRIES,
                false, 2);
        cosineDocumentResults = new HashMap<Integer, ArrayList<Integer>>();
        okapiDocumentResults = new HashMap<Integer, ArrayList<Integer>>();
        humanJudgement = CompareDocsUtil.readHumanJudgement("/Users/Eric/Desktop/Thesis/programs/src/labs/human_judgement.txt", 1, 3);
        System.out.println("Human Oracle Read");

        OkapiDistance okapiAlg = new OkapiDistance(documents);

        // For each query, compute the top 20 relevant documents for the okapi algorithm
        for (int id = 1; id <= queries.getSize(); id++) {
            QueryVector qv = (QueryVector) queries.getVectorById(id);
            okapiDocumentResults.put(id, qv.findClosestDocuments(documents, okapiAlg));
        }
        System.out.println("okapiDocumentResults computed");

        // Normalize vectors for cosine distance function
        documents.normalize(documents);
        queries.normalize(documents);

        CosineDistance cosineAlg = new CosineDistance();
        // For each query, compute the top 20 relevant documents for the cosine algorithm
        for (int id = 1; id <= queries.getSize(); id++) {
            QueryVector qv = (QueryVector) queries.getVectorById(id);
            cosineDocumentResults.put(id, qv.findClosestDocuments(documents, cosineAlg));
        }
        System.out.println("cosineDocumentResults computed");

        // Compare the difference between the two similarity algorithms
        System.out.println("Cosine MAP = " + CompareDocsUtil.computeMAP(humanJudgement, cosineDocumentResults, 20));
        System.out.println("Okapi MAP = " + CompareDocsUtil.computeMAP(humanJudgement, okapiDocumentResults, 20));

    }


}
