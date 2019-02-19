package DocumentClasses;

import java.io.Serializable;
import java.util.*;

/**
 * A generic description for a collection of terms.
 * Maps terms to frequencies.
 *
 * @author ericlabouve
 */
public abstract class TextVector implements Serializable {

    /** Stores the frequency for each word */
    private HashMap<String, Integer> termToFreqMap;

    /** Raw text of queries are needed for TermProxDistance functions */
    private ArrayList<String> rawText;

    /** Id of this vector */
    private int id;

    public TextVector() {
        termToFreqMap = new HashMap<String, Integer>();
        rawText = new ArrayList<String>();
    }

    /**
     * Will return the entry set of the normalizedTermToFreqMap.
     * @return the entry set of the normalizedTermToFreqMap.
     */
    public abstract Set<Map.Entry<String, Double>> getNormalizedVectorEntrySet();

    /**
     * Will normalize the frequency of each term using the TF-IDF formula
     * and store the results in normalizedTermToFreqMap
     * @param vc The database of vectors to normalize termToFreqMap against.
     */
    public abstract void normalize(VectorCollection vc);

    /**
     * Will return the normalized frequency of the word.
     * @param word The term to get the normalized frequency of.
     * @return the normalized frequency of word.
     */
    public abstract double getNormalizedFrequency(String word);





    public void setId(int id) {
        this.id = id;
    }

    public int getId() {
        return id;
    }

    public void appendToRawText(String term) {
        rawText.add(term);
    }

    public ArrayList<String> getRawText() {
        return rawText;
    }

    /**
     * Performs a logarithmic operation with base 2.
     * @param n Number to take logarithm of.
     * @return Log Base 2 of n.
     */
    public double log2(double n) {
        return Math.log(n) / Math.log(2);
    }

    /**
     * Computes the square root of the sum of the squares of the normalized frequencies.
     * The method calls the getNormalizedFrequency method to get the normalized frequencies.
     * @return the square root of the sum of the squares of the normalized frequencies.
     */
    public double getNormalizedL2Norm() {
        double runningTotal = 0;
        for (String term : termToFreqMap.keySet()) {
            runningTotal += Math.pow(getNormalizedFrequency(term), 2);
        }
        return Math.sqrt(runningTotal);
    }

/****************************************  Cosine + Okapi ********************************************/

    /**
     * Tuple class used for pairing TextVectors with their associated distance value
     */
    private class DistanceValue_Id {
        public double distanceValue;
        public int id;
        public DistanceValue_Id(double d, int id) {
            distanceValue = d;
            this.id = id;
        }
        @Override
        public String toString() {
            return "(" + distanceValue + ", " + id + ")";
        }
    }

    /**
     * Computes the distances between the query (the implicit parameter) and each
     * document in the VectorCollection. Once distances are calculated, the ids of
     * the closest 20 documents are returned in order.
     * @param documents The collection of vectors to compare the query against
     * @param distanceAlg The distance algorithm used to compare the query against the
     *                    vector collection
     * @return the 20 closest documents as an ArrayList<Integer>
     */
    public ArrayList<Integer> findClosestDocuments(VectorCollection documents, DocumentDistance distanceAlg) {
        List<DistanceValue_Id> rankedDocuments = new ArrayList<DistanceValue_Id>();
        // Compute all the distances for each document to this query vector
        // ids for vectors start at 1
        for (int id = 1; id <= documents.getSize(); id++) {
            TextVector tv = documents.getVectorById(id);
            double distance = distanceAlg.findDistance(this, tv, documents);
            DistanceValue_Id dis_id = new DistanceValue_Id(distance, id);
            rankedDocuments.add(dis_id);
        }
        ArrayList<Integer> closestDocuments = getTopDocs(rankedDocuments);
        return closestDocuments;
    }


    private ArrayList<Integer> getTopDocs(List<DistanceValue_Id> rankedDocuments) {
        // Sort TextVectors based on their distance value
        // o1 and o2 are DistanceValue_Id objects
        Collections.sort(rankedDocuments, (o1, o2) -> {
            if (o1.distanceValue > o2.distanceValue) {
                return -1;
            } else if (o1.distanceValue < o2.distanceValue) {
                return 1;
            } else {
                return 0;
            }
        });
        // Store top 20 TextVector indexes in |closestDocuments|
        ArrayList<Integer> closestDocuments = new ArrayList<Integer>();
        for (int i = 0; i < 20; i++) {
            try {
                closestDocuments.add(rankedDocuments.get(i).id);
            } catch (IndexOutOfBoundsException ex) {
                break;
            }
        }
        return closestDocuments;
    }

/****************************************  2 Constraints ********************************************/

    /**
     * Find the closest 20 documents to this query using 2Constrains
     * @param documents The collection of vectors to compare the query against
     * @param maxValue The max distance allowed between a pair of words in our generated graph
     * @return the 20 closest documents as an ArrayList<Integer>
     */
    public ArrayList<Integer> findClosestDocuments_2Constrains(GraphGenerator generator, VectorCollection documents, int maxValue) {
        List<DistanceValue_Id> rankedDocuments = new ArrayList<DistanceValue_Id>();
        // Compute all the distances for each document to this query vector
        for (int id = 1; id <= documents.getSize(); id++) {
            GraphGenerator.WeightedGraph graph =
                    generator.generateTree((QueryVector) this, documents, id, maxValue);
            double distance = GraphReader.findDistance_2Constraints(graph);
            DistanceValue_Id dis_id = new DistanceValue_Id(distance, id);
            rankedDocuments.add(dis_id);
        }
        ArrayList<Integer> closestDocuments = getTopDocs(rankedDocuments);

        return closestDocuments;
    }


/********************************************  Scoring Functions ********************************************/

    /**
     * Find the closest 20 documents to this query using function1_BestPath
     * @param documents The collection of vectors to compare the query against
     * @param maxValue The max distance allowed between a pair of words in our generated graph
     * @param f Scoring function for edge and node weights
     * @return the 20 closest documents as an ArrayList<Integer>
     */
    public ArrayList<Integer> findClosestDocuments_function1_BestPath(
            GraphGenerator generator, VectorCollection documents, int maxValue, GraphReader.ScoringFunction f) {
        List<DistanceValue_Id> rankedDocuments = new ArrayList<DistanceValue_Id>();
        // Compute all the distances for each document to this query vector
        for (int id = 1; id <= documents.getSize(); id++) {
            GraphGenerator.WeightedGraph graph =
                    generator.generateTree((QueryVector) this, documents, id, maxValue);
            double distance = GraphReader.findDistance_function_BestPath(graph, f);
            DistanceValue_Id dis_id = new DistanceValue_Id(distance, id);
            rankedDocuments.add(dis_id);
        }
        ArrayList<Integer> closestDocuments = getTopDocs(rankedDocuments);

        return closestDocuments;
    }

    /**
     * Find the closest 20 documents to this query using function1_BestPath_SumNodes
     * @param documents The collection of vectors to compare the query against
     * @param maxValue The max distance allowed between a pair of words in our generated graph
     * @param f Scoring function for edge and node weights
     * @return the 20 closest documents as an ArrayList<Integer>
     */
    public ArrayList<Integer> findClosestDocuments_function1_BestPath_SumNodes(
            GraphGenerator generator, VectorCollection documents, int maxValue, GraphReader.ScoringFunction f) {
        List<DistanceValue_Id> rankedDocuments = new ArrayList<DistanceValue_Id>();
        // Compute all the distances for each document to this query vector
        for (int id = 1; id <= documents.getSize(); id++) {
            GraphGenerator.WeightedGraph graph =
                    generator.generateTree((QueryVector) this, documents, id, maxValue);
            double distance = GraphReader.findDistance_function_BestPath_SumNodes(graph, f);
            DistanceValue_Id dis_id = new DistanceValue_Id(distance, id);
            rankedDocuments.add(dis_id);
        }
        ArrayList<Integer> closestDocuments = getTopDocs(rankedDocuments);

        return closestDocuments;
    }

    /**
     * Find the closest 20 documents to this query using function2_BestPath
     * @param documents The collection of vectors to compare the query against
     * @param maxValue The max distance allowed between a pair of words in our generated graph
     * @param _f Scoring function for edge and node weights
     * @return the 20 closest documents as an ArrayList<Integer>
     */
    public ArrayList<Integer> findClosestDocuments_function2_BestPath(
            GraphGenerator generator, VectorCollection documents, int maxValue, GraphReader.ScoringFunction _f) {
        List<DistanceValue_Id> rankedDocuments = new ArrayList<DistanceValue_Id>();
        GraphReader.ScoringFunction2 f = (GraphReader.ScoringFunction2) _f;
        // Compute all the distances for each document to this query vector
        for (int id = 1; id <= documents.getSize(); id++) {
            f.addCurDocId(id);
            GraphGenerator.WeightedGraph graph =
                    generator.generateTree((QueryVector) this, documents, id, maxValue);
            double distance = GraphReader.findDistance_function_BestPath(graph, f);
            DistanceValue_Id dis_id = new DistanceValue_Id(distance, id);
            rankedDocuments.add(dis_id);
        }
        ArrayList<Integer> closestDocuments = getTopDocs(rankedDocuments);

        return closestDocuments;
    }

    /**
     * Find the closest 20 documents to this query using function2_BestPath_SumNodes
     * @param documents The collection of vectors to compare the query against
     * @param maxValue The max distance allowed between a pair of words in our generated graph
     * @param _f Scoring function for edge and node weights
     * @return the 20 closest documents as an ArrayList<Integer>
     */
    public ArrayList<Integer> findClosestDocuments_function2_BestPath_SumNodes(
            GraphGenerator generator, VectorCollection documents, int maxValue, GraphReader.ScoringFunction _f) {
        List<DistanceValue_Id> rankedDocuments = new ArrayList<DistanceValue_Id>();
        GraphReader.ScoringFunction2 f = (GraphReader.ScoringFunction2) _f;
        // Compute all the distances for each document to this query vector
        for (int id = 1; id <= documents.getSize(); id++) {
            f.addCurDocId(id);
            GraphGenerator.WeightedGraph graph =
                    generator.generateTree((QueryVector) this, documents, id, maxValue);
            double distance = GraphReader.findDistance_function_BestPath_SumNodes(graph, f);
            DistanceValue_Id dis_id = new DistanceValue_Id(distance, id);
            rankedDocuments.add(dis_id);
        }
        ArrayList<Integer> closestDocuments = getTopDocs(rankedDocuments);

        return closestDocuments;
    }

    /**
     * Find the closest 20 documents to this query using function2_BestPath_SumNodes
     * @param documents The collection of vectors to compare the query against
     * @param maxValue The max distance allowed between a pair of words in our generated graph
     * @param _f Scoring function for edge and node weights
     * @return the 20 closest documents as an ArrayList<Integer>
     */
    public ArrayList<Integer> findClosestDocuments_function2_BestPath_SumNodes_Cosine(
            GraphGenerator generator, VectorCollection documents, int maxValue, GraphReader.ScoringFunction _f) {
        CosineDistance cosineAlg = new CosineDistance();
        List<DistanceValue_Id> rankedDocuments = new ArrayList<DistanceValue_Id>();
        GraphReader.ScoringFunction2 f = (GraphReader.ScoringFunction2) _f;
        // Compute all the distances for each document to this query vector
        for (int id = 1; id <= documents.getSize(); id++) {
            TextVector tv = documents.getVectorById(id);
            f.addCurDocId(id);
            GraphGenerator.WeightedGraph graph =
                    generator.generateTree((QueryVector) this, documents, id, maxValue);
            double distance = GraphReader.findDistance_function_BestPath_SumNodes(graph, f);
            double cosineDistance = cosineAlg.findDistance(this, tv, documents);
            DistanceValue_Id dis_id = new DistanceValue_Id(distance * cosineDistance, id);
            rankedDocuments.add(dis_id);
        }
        ArrayList<Integer> closestDocuments = getTopDocs(rankedDocuments);

        return closestDocuments;
    }

    /**
     * Find the closest 20 documents to this query using function2_BestPath_SumNodes
     * @param documents The collection of vectors to compare the query against
     * @param maxValue The max distance allowed between a pair of words in our generated graph
     * @param _f Scoring function for edge and node weights
     * @return the 20 closest documents as an ArrayList<Integer>
     */
    public ArrayList<Integer> findClosestDocuments_function3_BestPath_SumNodes(
            GraphGenerator generator, VectorCollection documents, int maxValue, GraphReader.ScoringFunction _f) {
        List<DistanceValue_Id> rankedDocuments = new ArrayList<DistanceValue_Id>();
        GraphReader.ScoringFunction3 f = (GraphReader.ScoringFunction3) _f;
        // Compute all the distances for each document to this query vector
        for (int id = 1; id <= documents.getSize(); id++) {
            f.addCurDocId(id);
            GraphGenerator.WeightedGraph graph =
                    generator.generateTree((QueryVector) this, documents, id, maxValue);
            double distance = GraphReader.findDistance_function_BestPath_SumNodes(graph, f);
            DistanceValue_Id dis_id = new DistanceValue_Id(distance, id);
            rankedDocuments.add(dis_id);
        }
        ArrayList<Integer> closestDocuments = getTopDocs(rankedDocuments);

        return closestDocuments;
    }

    /**
     * Find the closest 20 documents to this query using function2_SumEdges
     * @param documents The collection of vectors to compare the query against
     * @param maxValue The max distance allowed between a pair of words in our generated graph
     * @param _f Scoring function for edge and node weights
     * @return the 20 closest documents as an ArrayList<Integer>
     */
    public ArrayList<Integer> findClosestDocuments_function2_SumEdges(
            GraphGenerator generator, VectorCollection documents, int maxValue, GraphReader.ScoringFunction _f) {
        List<DistanceValue_Id> rankedDocuments = new ArrayList<DistanceValue_Id>();
        GraphReader.ScoringFunction2 f = (GraphReader.ScoringFunction2) _f;
        // Compute all the distances for each document to this query vector
        for (int id = 1; id <= documents.getSize(); id++) {
            f.addCurDocId(id);
            GraphGenerator.WeightedGraph graph =
                    generator.generateTree((QueryVector) this, documents, id, maxValue);
            double distance = GraphReader.findDistance_function_SumEdges(graph, f);
            DistanceValue_Id dis_id = new DistanceValue_Id(distance, id);
            rankedDocuments.add(dis_id);
        }
        ArrayList<Integer> closestDocuments = getTopDocs(rankedDocuments);
        return closestDocuments;
    }

    /**
     * Find the closest 20 documents to this query using function2_SumEdges_BestPath
     * @param documents The collection of vectors to compare the query against
     * @param maxValue The max distance allowed between a pair of words in our generated graph
     * @param _f Scoring function for edge and node weights
     * @return the 20 closest documents as an ArrayList<Integer>
     */
    public ArrayList<Integer> findClosestDocuments_function2_SumEdges_BestPath(
            GraphGenerator generator, VectorCollection documents, int maxValue, GraphReader.ScoringFunction _f) {
        List<DistanceValue_Id> rankedDocuments = new ArrayList<DistanceValue_Id>();
        GraphReader.ScoringFunction2 f = (GraphReader.ScoringFunction2) _f;
        // Compute all the distances for each document to this query vector
        for (int id = 1; id <= documents.getSize(); id++) {
            f.addCurDocId(id);
            GraphGenerator.WeightedGraph graph =
                    generator.generateTree((QueryVector) this, documents, id, maxValue);
            double distance = GraphReader.findDistance_function_SumEdges_BestPath(graph, f);
            DistanceValue_Id dis_id = new DistanceValue_Id(distance, id);
            rankedDocuments.add(dis_id);
        }
        ArrayList<Integer> closestDocuments = getTopDocs(rankedDocuments);
        return closestDocuments;
    }

    /**
     * Find the closest 20 documents to this query using function2_SumEdges_BestPath_SumNodes
     * @param documents The collection of vectors to compare the query against
     * @param maxValue The max distance allowed between a pair of words in our generated graph
     * @param _f Scoring function for edge and node weights
     * @return the 20 closest documents as an ArrayList<Integer>
     */
    public ArrayList<Integer> findClosestDocuments_function2_SumEdges_BestPath_SumNodes(
            GraphGenerator generator, VectorCollection documents, int maxValue, GraphReader.ScoringFunction _f) {
        List<DistanceValue_Id> rankedDocuments = new ArrayList<DistanceValue_Id>();
        GraphReader.ScoringFunction2 f = (GraphReader.ScoringFunction2) _f;
        // Compute all the distances for each document to this query vector
        for (int id = 1; id <= documents.getSize(); id++) {
            f.addCurDocId(id);
            GraphGenerator.WeightedGraph graph =
                    generator.generateTree((QueryVector) this, documents, id, maxValue);
            double distance = GraphReader.findDistance_function_SumEdges_BestPath_SumNodes(graph, f);
            DistanceValue_Id dis_id = new DistanceValue_Id(distance, id);
            rankedDocuments.add(dis_id);
        }
        ArrayList<Integer> closestDocuments = getTopDocs(rankedDocuments);
        return closestDocuments;
    }

/*******************************************************************************************************/

    /**
     * Will return the entry set of the termToFreqMap.
     * @return the entry set of the termToFreqMap.
     */
    public Set<Map.Entry<String, Integer>> getRawVectorEntrySet() {
        return termToFreqMap.entrySet();
    }


    /**
     * Adds a word to the termToFreqMap. If the word is not new, the frequency is incremented by one.
     * @param word The word to add to the termToFreqMap.
     */
    public void add(String word) {
        if (termToFreqMap.containsKey(word)) {
            termToFreqMap.put(word, termToFreqMap.get(word) + 1);
        }
        else {
            termToFreqMap.put(word, 1);
        }
    }

    /**
     * @return true if the word is in the termToFreqMap and false otherwise.
     */
    public boolean contains(String word) {
        return termToFreqMap.containsKey(word);
    }

    /**
     * @return the frequency of the word.
     */
    public int getRawFrequency(String word) {
        Integer count = termToFreqMap.get(word);
        return count != null ? count : 0;
    }

    /**
     * @Return the total number of words that are stored for the document
     */
    public int getTotalWordCount() {
        int runningTotal = 0;
        for (Map.Entry<String, Integer> pair : termToFreqMap.entrySet()) {
            runningTotal += pair.getValue();
        }
        return runningTotal;
    }

    /**
     * @return the number of distinct words that are stored
     */
    public int getDistinctWordCount() {
        return termToFreqMap.size();
    }

    /**
     * @return the highest word frequency
     */
    public int getHighestRawFrequency() {
        int curHighest = 0;
        for (Map.Entry<String, Integer> pair : getRawVectorEntrySet()) {
            if (pair.getValue() > curHighest) {
                curHighest = pair.getValue();
            }
        }
        return curHighest;
    }

    /**
     * @return the word with the highest frequency.
     */
    public String getMostFrequentWord() {
        int curHighest = 0;
        String word = "";
        for (Map.Entry<String, Integer> pair : getRawVectorEntrySet()) {
            if (pair.getValue() > curHighest) {
                curHighest = pair.getValue();
                word = pair.getKey();
            }
        }
        return word;
    }
}
