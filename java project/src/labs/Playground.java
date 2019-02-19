package labs;

import DocumentClasses.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

/**
 * Created by Eric on 1/25/18.
 *
 * @author ericlabouve
 */
public class Playground {

    /** The Documents */
    public static VectorCollection documents;
    /** The Queries */
    public static VectorCollection queries;
    /** Contains top 20 relevant documents in order for each query according to Consine Similarity */
    public static HashMap<Integer, ArrayList<Integer>> cosineDocumentResults;
    /** The variable contains the IDs of the relevant documents for each query */
    public static HashMap<Integer, ArrayList<Integer>> humanJudgement;

    public static void main(String[] args) {
        documents = new VectorCollection("/Users/Eric/Desktop/Thesis/programs/datasets/cran/cran.all.1400",
                VectorCollection.VectorType.DOCUMENTS,
                true, 1);
        queries = new VectorCollection("/Users/Eric/Desktop/Thesis/programs/datasets/cran/cran.qry",
                VectorCollection.VectorType.QUIRIES,
                true, 1);

        cosineDocumentResults = new HashMap<Integer, ArrayList<Integer>>();
        humanJudgement = new HashMap<Integer, ArrayList<Integer>>();

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

        // Read in the human judgement oracle
        // Format: query number, document number, and degree of relevance
        try(BufferedReader br = new BufferedReader(new FileReader("/Users/Eric/Desktop/Thesis/programs/datasets/cran/cranqrel"))) {
            String line = br.readLine();
            // Continue until end of file is reached
            while (line != null) {
                String[] tokens = line.split("\\s+");
                int queryNumber = Integer.parseInt(tokens[0]);
                int docNumber = Integer.parseInt(tokens[1]);
                int degreeRel = Integer.parseInt(tokens[2]);
                // Check to see if we have seen this query number before
                if (!humanJudgement.containsKey(queryNumber)) {
                    humanJudgement.put(queryNumber, new ArrayList<Integer>());
                }
                // Check to see if the docNumber is relevant CAN TIGHTEN THESE BOUNDS
                if (degreeRel >= 1 && degreeRel <= 3) {
                    humanJudgement.get(queryNumber).add(docNumber);
                }
                line = br.readLine();
            }
        } catch (java.io.FileNotFoundException ex) {
            ex.printStackTrace();
        } catch (java.io.IOException ex) {
            ex.printStackTrace();
        }
        System.out.println("Human Oracle Read");

        // Compare the difference between the two similarity algorithms
        System.out.println("Cosine MAP = " + CompareDocsUtil.computeMAP(humanJudgement, cosineDocumentResults, 20));
    }
}
