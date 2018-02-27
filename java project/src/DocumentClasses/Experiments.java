package DocumentClasses;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * @author ericlabouve
 */
public class Experiments {

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

    /** Contains top 20 relevant documents in order for each query according to a specific ditance function*/
    public static HashMap<Integer, ArrayList<Integer>> function3_BestPath_SumNodes_DocumentResults;
    public static HashMap<Integer, ArrayList<Integer>> function2_SumEdges_BestPath_SumNodes_DocumentResults;
    public static HashMap<Integer, ArrayList<Integer>> function2_SumEdges_BestPath_DocumentResults;
    public static HashMap<Integer, ArrayList<Integer>> function2_SumEdges_DocumentResults;
    public static HashMap<Integer, ArrayList<Integer>> function2_BestPath_DocumentResults;
    public static HashMap<Integer, ArrayList<Integer>> function2_BestPath_SumNodes_DocumentResults;
    public static HashMap<Integer, ArrayList<Integer>> function1_BestPath_DocumentResults;
    public static HashMap<Integer, ArrayList<Integer>> function1_BestPath_SumNodes_DocumentResults;
    public static HashMap<Integer, ArrayList<Integer>> twoConstraintsDocumentResults;
    public static HashMap<Integer, ArrayList<Integer>> function2_BestPath_SumNodes_Cosine_DocumentResults;


    public static GraphGenerator generator;

    /** Number of queries to evaluate */
    public static int numQueries;

    /**
     * Load data from your binary file.
     * Next, initialize the queries variable, and call the normalize() method on both variables.
     * Finally, print the 20 most relevant documents for each query.
     * @param args
     */
    public static void main(String[] args) {

        /***** Algorithm scores to beat *****/

        // Instantiate and fill out all variables
        documents = new VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.all.1400",
                VectorCollection.VectorType.DOCUMENTS,
                false, 2);
        queries = new VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.qry",
                VectorCollection.VectorType.QUIRIES,
                false, 2);
        humanJudgement = CompareDocsUtil.readHumanJudgement("/Users/Eric/Desktop/Thesis/projects/java project/src/labs/human_judgement.txt", 1, 3);
        cosineDocumentResults = new HashMap<Integer, ArrayList<Integer>>();
        okapiDocumentResults = new HashMap<Integer, ArrayList<Integer>>();

        twoConstraintsDocumentResults = new HashMap<>();
        function1_BestPath_DocumentResults = new HashMap<>();
        function1_BestPath_SumNodes_DocumentResults = new HashMap<>();
        function2_BestPath_DocumentResults = new HashMap<>();
        function2_BestPath_SumNodes_DocumentResults = new HashMap<>();
        function2_SumEdges_DocumentResults = new HashMap<>();
        function2_SumEdges_BestPath_DocumentResults = new HashMap<>();
        function2_SumEdges_BestPath_SumNodes_DocumentResults = new HashMap<>();
        function3_BestPath_SumNodes_DocumentResults = new HashMap<>();
        function2_BestPath_SumNodes_Cosine_DocumentResults = new HashMap<>();

        // Normalize vectors for node weights and cosine algorithm
        documents.normalize(documents);
        queries.normalize(documents);

        generator = new GraphGenerator();
        numQueries = 225;

        // Compute Distance Scores
   //     computeOkapiScores();
        computeCosineScores();
//        compute_2Constraint_Scores();
//        compute_function1_BestPath_Scores();
//        compute_function1_BestPath_SumNodes_Scores();
//        compute_function2_BestPath_Scores();
      //  compute_function2_BestPath_SumNodes_Scores();
//        compute_function2_SumEdges_Score();
//        compute_function2_SumEdges_BestPath_Scores();
//        compute_function2_SumEdges_BestPath_SumNodes_Scores();
     //   compute_function3_BestPath_SumNodes_Scores();
     //   compute_function2_BestPath_SumNodes_Cosine_Scores();


        /***** Experiemental Section *****/
/*
        System.out.println("Normalized Values:");
        for (String term : queries.getVectorById(1).getRawText()) {
            System.out.println(term + ":" + queries.getVectorById(1).getNormalizedFrequency(term) + " ");
        }

        System.out.println("Q1, D14");
        GraphGenerator.WeightedGraph graph =
                generator.generateTree((QueryVector) queries.getVectorById(1), documents, 14, Integer.MAX_VALUE);
        System.out.println(graph.toString());

        System.out.println("\nQ1, D378");
        GraphGenerator.WeightedGraph graph2 =
                generator.generateTree((QueryVector) queries.getVectorById(1), documents, 378, Integer.MAX_VALUE);
        System.out.println(graph2.toString());

*/

    }

    public static void compute_function2_BestPath_SumNodes_Cosine_Scores() {
        GraphReader.ScoringFunction f = new GraphReader.ScoringFunction2(3, Double.MAX_VALUE, 1, documents);
        for (int id = 1; id <= queries.getSize(); id++) {
            QueryVector qv = (QueryVector) queries.getVectorById(id);
            function2_BestPath_SumNodes_Cosine_DocumentResults.put(id,
                    qv.findClosestDocuments_function2_BestPath_SumNodes_Cosine(
                            generator, documents, Integer.MAX_VALUE, f));
        }
        double mapScore = CompareDocsUtil.computeMAP(humanJudgement, function2_BestPath_SumNodes_Cosine_DocumentResults, numQueries);
        System.out.println("function2 Best Path + Sum Nodes + Cosine MAP = " + mapScore);
    }

    /**
     *
     */
    private static void compute_function3_BestPath_SumNodes_Scores() {
        double bestMapScore = 0;
        double bestAlpha = 1;
        double bestBeta = 1;
        for (double alpha = 1; alpha <= 10; alpha += 1) {
            System.out.print(alpha);
            for (double beta = 10; beta <= 30; beta += 1) {
                GraphReader.ScoringFunction f = new GraphReader.ScoringFunction3(alpha, beta, 2, documents);
                for (int id = 1; id <= queries.getSize(); id++) {
                    QueryVector qv = (QueryVector) queries.getVectorById(id);

                    /** To achieve different scores we can:
                     Vary maxValue to limit the graph size
                     Vary alpha to change function's edge score
                     Vary beta to change function's node score   */
                    function3_BestPath_SumNodes_DocumentResults.put(id,
                            qv.findClosestDocuments_function3_BestPath_SumNodes(
                                    generator, documents, Integer.MAX_VALUE, f));
                }
                double mapScore = CompareDocsUtil.computeMAP(humanJudgement, function3_BestPath_SumNodes_DocumentResults, numQueries);
                if (mapScore > bestMapScore) {
                    bestMapScore = mapScore;
                    bestAlpha = alpha;
                    bestBeta = beta;
                }
            }
        }
        System.out.println("\nfunction3 Best Path + Sum Nodes MAP = " + bestMapScore + ", alpha = " + bestAlpha + ", beta = " + bestBeta);
        System.out.println(function3_BestPath_SumNodes_DocumentResults.get(1));
    }

    /**
     * function2 Sum Edges + Best Path + Sum Nodes MAP = 0.22598575615820007, alpha = 4.0, beta = 10.0
     */
    private static void compute_function2_SumEdges_BestPath_SumNodes_Scores() {
        double bestMapScore = 0;
        double bestAlpha = 1;
        double bestBeta = 1;
        for (double alpha = 3; alpha <= 3; alpha += 1) {
            for (double beta = 30; beta <= 30; beta += 1) {
                GraphReader.ScoringFunction f = new GraphReader.ScoringFunction2(alpha, beta, documents);
                for (int id = 1; id <= queries.getSize(); id++) {
                    QueryVector qv = (QueryVector) queries.getVectorById(id);

                    /** To achieve different scores we can:
                            Vary maxValue to limit the graph size
                            Vary alpha to change function's edge score
                            Vary beta to change function's node score   */
                    function2_SumEdges_BestPath_SumNodes_DocumentResults.put(id,
                            qv.findClosestDocuments_function2_SumEdges_BestPath_SumNodes(
                                    generator, documents, Integer.MAX_VALUE, f));
                }
                double mapScore = CompareDocsUtil.computeMAP(humanJudgement, function2_SumEdges_BestPath_SumNodes_DocumentResults, numQueries);
                if (mapScore > bestMapScore) {
                    bestMapScore = mapScore;
                    bestAlpha = alpha;
                    bestBeta = beta;
                }
            }
        }
        System.out.println("function2 Sum Edges + Best Path + Sum Nodes MAP = " + bestMapScore + ", alpha = " + bestAlpha + ", beta = " + bestBeta);
    }

    /**
     * function2 Sum Edges + Best Path MAP = 0.22600529149758838, alpha = 3.0, beta = 10.0
     */
    private static void compute_function2_SumEdges_BestPath_Scores() {
        double bestMapScore = 0;
        double bestAlpha = 1;
        double bestBeta = 1;
        for (double alpha = 1; alpha <= 10; alpha += 1) {
            for (double beta = 10; beta <= 40; beta += 1) {
                GraphReader.ScoringFunction f = new GraphReader.ScoringFunction2(alpha, beta, documents);
                for (int id = 1; id <= queries.getSize(); id++) {
                    QueryVector qv = (QueryVector) queries.getVectorById(id);

                    /** To achieve different scores we can:
                            Vary maxValue to limit the graph size
                            Vary alpha to change function's edge score
                            Vary beta to change function's node score   */
                    function2_SumEdges_BestPath_DocumentResults.put(id,
                            qv.findClosestDocuments_function2_SumEdges_BestPath(
                                    generator, documents, Integer.MAX_VALUE, f));
                }
                double mapScore = CompareDocsUtil.computeMAP(humanJudgement, function2_SumEdges_BestPath_DocumentResults, numQueries);
                if (mapScore > bestMapScore) {
                    bestMapScore = mapScore;
                    bestAlpha = alpha;
                    bestBeta = beta;
                }
            }
        }
        System.out.println("function2 Sum Edges + Best Path MAP = " + bestMapScore + ", alpha = " + bestAlpha + ", beta = " + bestBeta);
    }

    /**
     * function2 Sum Edges MAP = 0.213341432435445, alpha = 8.0
     */
    private static void compute_function2_SumEdges_Score() {
        double bestMapScore = 0;
        double bestAlpha = 1;
        for (double alpha = 1; alpha <= 10; alpha += 1) {
            GraphReader.ScoringFunction f = new GraphReader.ScoringFunction2(alpha, documents);
            for (int id = 1; id <= queries.getSize(); id++) {
                QueryVector qv = (QueryVector) queries.getVectorById(id);
                /** To achieve different scores we can:
                        Vary maxValue to limit the graph size
                        Vary alpha to change function's edge score   */
                function2_SumEdges_DocumentResults.put(id,
                        qv.findClosestDocuments_function2_SumEdges(generator, documents, Integer.MAX_VALUE, f));
            }
            double mapScore = CompareDocsUtil.computeMAP(humanJudgement, function2_SumEdges_DocumentResults, numQueries);
            if (mapScore > bestMapScore) {
                bestMapScore = mapScore;
                bestAlpha = alpha;
            }
        }
        System.out.println("function2 Sum Edges MAP = " + bestMapScore + ", alpha = " + bestAlpha);
    }

    /**
     * function2 Best Path + Sum Nodes MAP = 0.3127619749657657, alpha = 3.0, beta = 30.0
     */
    private static void compute_function2_BestPath_SumNodes_Scores() {
        double bestMapScore = 0;
        double bestAlpha = 1;
        double bestBeta = 1;
        double bestGamma = 1;
        for (double alpha = 3; alpha <= 3; alpha += 1) {
            for (double beta = 30; beta <= 30; beta += 1) {
                for (double gamma = 1; gamma <= 1; gamma += 0.1) {
                    GraphReader.ScoringFunction f = new GraphReader.ScoringFunction2(alpha, beta, gamma, documents);
                    for (int id = 1; id <= queries.getSize(); id++) {
                        QueryVector qv = (QueryVector) queries.getVectorById(id);

                        /** To achieve different scores we can:
                         Vary maxValue to limit the graph size
                         Vary alpha to change function's edge score
                         Vary beta to change function's node score   */
                        function2_BestPath_SumNodes_DocumentResults.put(id,
                                qv.findClosestDocuments_function2_BestPath_SumNodes(
                                        generator, documents, Integer.MAX_VALUE, f));
                    }
                    double mapScore = CompareDocsUtil.computeMAP(humanJudgement, function2_BestPath_SumNodes_DocumentResults, numQueries);
                    if (mapScore > bestMapScore) {
                        bestMapScore = mapScore;
                        bestAlpha = alpha;
                        bestBeta = beta;
                        bestGamma = gamma;
                    }
                }
            }
        }
        System.out.println("function2 Best Path + Sum Nodes MAP = " + bestMapScore + ", a=" + bestAlpha + ", b=" + bestBeta + ", g=" + bestGamma);
        System.out.println(function2_BestPath_SumNodes_DocumentResults.get(1));
    }

    /**
     * function2 Best Path MAP = 0.27432051295261395, alpha = 5.0
     */
    private static void compute_function2_BestPath_Scores() {
        double bestMapScore = 0;
        double bestAlpha = 1;
        for (double alpha = 1; alpha <= 10; alpha += 1) {
            GraphReader.ScoringFunction f = new GraphReader.ScoringFunction2(alpha, documents);
            for (int id = 1; id <= queries.getSize(); id++) {
                QueryVector qv = (QueryVector) queries.getVectorById(id);
                /** To achieve different scores we can:
                        Vary maxValue to limit the graph size
                        Vary alpha to change function's edge score   */
                function2_BestPath_DocumentResults.put(id,
                        qv.findClosestDocuments_function2_BestPath(generator, documents, Integer.MAX_VALUE, f));
            }
            double mapScore = CompareDocsUtil.computeMAP(humanJudgement, function2_BestPath_DocumentResults, numQueries);
            if (mapScore > bestMapScore) {
                bestMapScore = mapScore;
                bestAlpha = alpha;
            }
        }
        System.out.println("function2 Best Path MAP = " + bestMapScore + ", alpha = " + bestAlpha);
    }

    /**
     * function1 Best Path + Sum Nodes MAP = 0.25284019221519216, alpha = 3.0, beta = 22.0
     */
    private static void compute_function1_BestPath_SumNodes_Scores() {
        double bestMapScore = 0;
        double bestAlpha = 1;
        double bestBeta = 1;
        for (double alpha = 3; alpha <= 3; alpha += 1) {
            for (double beta = 22; beta <= 22; beta += 1) {
                GraphReader.ScoringFunction f = new GraphReader.ScoringFunction1(alpha, beta);
                for (int id = 1; id <= queries.getSize(); id++) {
                    QueryVector qv = (QueryVector) queries.getVectorById(id);

                    /** To achieve different scores we can:
                            Vary maxValue to limit the graph size
                            Vary alpha to change function's edge score
                            Vary beta to change function's node score   */
                    function1_BestPath_SumNodes_DocumentResults.put(id,
                            qv.findClosestDocuments_function1_BestPath_SumNodes(
                                    generator, documents, Integer.MAX_VALUE, f));
                }
                double mapScore = CompareDocsUtil.computeMAP(humanJudgement, function1_BestPath_SumNodes_DocumentResults, numQueries);
                if (mapScore > bestMapScore) {
                    bestMapScore = mapScore;
                    bestAlpha = alpha;
                    bestBeta = beta;
                }
            }
        }
        System.out.println("function1 Best Path + Sum Nodes MAP = " + bestMapScore + ", alpha = " + bestAlpha + ", beta = " + bestBeta);
    }

    /**
     * function1 Best Path MAP = 0.22510087095710155, alpha = 3.0
     */
    private static void compute_function1_BestPath_Scores() {
        double bestMapScore = 0;
        double bestAlpha = 1;

        for (double alpha = 1; alpha <= 10; alpha += 1) {
            GraphReader.ScoringFunction f = new GraphReader.ScoringFunction1(alpha);
            for (int id = 1; id <= queries.getSize(); id++) {
                QueryVector qv = (QueryVector) queries.getVectorById(id);

                /** To achieve different scores we can:
                        Vary maxValue to limit the graph size
                        Vary alpha to change function's edge score   */
                function1_BestPath_DocumentResults.put(id,
                        qv.findClosestDocuments_function1_BestPath(generator, documents, Integer.MAX_VALUE, f));
            }
            double mapScore = CompareDocsUtil.computeMAP(humanJudgement, function1_BestPath_DocumentResults, numQueries);
            if (mapScore > bestMapScore) {
                bestMapScore = mapScore;
                bestAlpha = alpha;
            }
        }
        System.out.println("function1 Best Path MAP = " + bestMapScore + ", alpha = " + bestAlpha);
    }

    /**
     * 2Constraint MAP = 0.06186449579831933
     */
    private static void compute_2Constraint_Scores() {
        // For each query, compute the top 20 relevant documents for the 2Constraint algorithm
        for (int id = 1; id <= queries.getSize(); id++) {
            QueryVector qv = (QueryVector) queries.getVectorById(id);

            /** To achieve different scores we can:
                    Vary maxValue to limit the graph size                    */
            twoConstraintsDocumentResults.put(id,
                    qv.findClosestDocuments_2Constrains(generator, documents, Integer.MAX_VALUE));
        }
        System.out.println("2Constraint MAP = " + CompareDocsUtil.computeMAP(humanJudgement, twoConstraintsDocumentResults, numQueries));
    }

    /**
     * Cosine MAP = 0.29881118219108416
     */
    private static void computeCosineScores() {
        CosineDistance cosineAlg = new CosineDistance();
        // For each query, compute the top 20 relevant documents for the cosine algorithm
        for (int id = 1; id <= queries.getSize(); id++) {
            QueryVector qv = (QueryVector) queries.getVectorById(id);
            cosineDocumentResults.put(id, qv.findClosestDocuments(documents, cosineAlg));
        }
        System.out.println("Cosine MAP = " + CompareDocsUtil.computeMAP(humanJudgement, cosineDocumentResults, numQueries));
        System.out.println(cosineDocumentResults.get(1));
        System.out.println(cosineDocumentResults.get(2));
    }

    /**
     * Okapi MAP = 0.29305451674068533
     */
    private static void computeOkapiScores() {
        OkapiDistance okapiAlg = new OkapiDistance(documents);
        // For each query, compute the top 20 relevant documents for the okapi algorithm
        for (int id = 1; id <= queries.getSize(); id++) {
            QueryVector qv = (QueryVector) queries.getVectorById(id);
            okapiDocumentResults.put(id, qv.findClosestDocuments(documents, okapiAlg));
        }
        System.out.println("Okapi MAP = " + CompareDocsUtil.computeMAP(humanJudgement, okapiDocumentResults, numQueries));
        System.out.println(okapiDocumentResults.get(1));
    }
}
