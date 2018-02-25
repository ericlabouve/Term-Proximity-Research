package DocumentClasses;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * @author ericlabouve
 */
public class CompareDocsUtil {

    /**
     * @param path File path
     * @param best Beginning range for human judgement
     * @param best Ending range for human judgement
     * @return Mapping from Queries to a list of related documents
     */
    public static HashMap<Integer, ArrayList<Integer>> readHumanJudgement(String path, int best, int worst) {
        HashMap<Integer, ArrayList<Integer>> humanJudgement = new HashMap<Integer, ArrayList<Integer>>();
        try(BufferedReader br = new BufferedReader(new FileReader(path))) {
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
                if (degreeRel >= best && degreeRel <= worst) {
                    humanJudgement.get(queryNumber).add(docNumber);
                }
                line = br.readLine();
            }
        } catch (java.io.FileNotFoundException ex) {
            ex.printStackTrace();
        } catch (java.io.IOException ex) {
            ex.printStackTrace();
        }
        return humanJudgement;
    }

    /**
     * Computes the MAP score between the two input hash maps using only the first 20 queries
     *
     * Mean Average Precision is computed by taking the precision in order from start to end of the
     * |documentResults| list. Look up the algorithm online because it is hard to explain in a short
     * text above this method.
     * @param humanJudgement Contains the oracle
     * @param documentResults Contains the results from the distance algorithm.
     * @param numQueries The number of queries to compute the MAP score over
     * @return
     */
    public static double computeMAP(HashMap<Integer, ArrayList<Integer>> humanJudgement,
                                    HashMap<Integer, ArrayList<Integer>> documentResults,
                                    int numQueries) {
        double totalSum = 0;
        // For the first 20 queries that have relevant documents
        int queryNumber = 1;
        int numQueriesEvaluated = 0;
        //
        while (numQueriesEvaluated < numQueries && humanJudgement.get(queryNumber) != null) {
            // If we have relevant documents for the query
            if (humanJudgement.get(queryNumber).size() > 0) {
                double precisionSum = 0;
                // Get the first 20 relevant documents for this query result
                ArrayList<Integer> predictedRelDocs = documentResults.get(queryNumber);
                // For each of the 20 predicted relevant document
                for (int docNumber = 1; docNumber <= predictedRelDocs.size(); docNumber++) {
                    int predictedDoc = predictedRelDocs.get(docNumber - 1);
                    // If this predicted document appears in our human judgement oracle
                    if (isRelevant(predictedDoc, humanJudgement.get(queryNumber))) {
                        double numRelevantReturned = intersection(predictedRelDocs.subList(0, docNumber), humanJudgement.get(queryNumber));
                        precisionSum += precision(numRelevantReturned, docNumber);
                    }
                }
                precisionSum /= humanJudgement.get(queryNumber).size();
                totalSum += precisionSum;
                numQueriesEvaluated++;
            }
            queryNumber++;
        }
        //System.out.print("Number Queries Evaluated = " + numQueriesEvaluated + ", ");
        return totalSum / numQueriesEvaluated;
    }

    public static double precision(double numRelevantReturned, double totalReturned) {
        return numRelevantReturned / totalReturned;
    }

    public static boolean isRelevant(int docNum, ArrayList<Integer> oracle) {
        return oracle.contains(docNum);
    }

    public static double intersection(List<Integer> list1, List<Integer> list2) {
        double sum = 0;
        for (int i : list1)
            for (int j : list2)
                if (i == j)
                    sum++;
        return sum;
    }
}
