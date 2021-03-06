package DocumentClasses;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Queue;

/**
 * Reads the graph generated by GraphGenerator and assigns a score
 *
 * @author ericlabouve
 */
public class GraphReader {


/************************************** 2 Constraints Best Path Algorithm ********************************************/

    /**
     We must first define what is considered the “best” substring. 
     We want to find a path through the G that satisfies two constraints:
        1. The path contains the most amount of query terms.
        2. No other path in the G contains an equal number of nodes with a shorter sum of distance weights.

     Two Constraints Algorithm:
     Idea: Traverse the G starting from the leaf nodes up to the root(s) and
            keep track of the path that satisfies constraints 1 and 2.
     Complexity: O(E + Number of Leaf Nodes) where E is the number of edges in the G

     G(V, E)	                        // G with vertices and edges
     Best = NULL                        // The root node containing the best discovered path
     V.edgePathLength = +infinity		// All vertices in G start with worst edge path length (default)
     V.nodePathLength = 0				// All vertices in G start with worst node path length (default)
     V.bestChild = Null                 // All vertices in G don't start with a best child (default)
     Q <— All leaf nodes in G

     While the Q is not empty:
        (node, child) = Q.dequeue()

        If node is a leaf node:
            node.edgePathLength = 0
            node.nodePathLength = 1

        // If new path has more query terms OR (path has same number of query terms AND is shorter)
        Else If (node.nodePathLength < child.nodePathLength + 1)
            OR (node.nodePathLength == child.nodePathLength + 1
            AND node.edgePathLength > child.edgePathLength + Edge(node, child) ):
            node.bestChild = child
            node.edgePathLength = child.edgePathLength + Edge(node, child)
            node.nodePathLength = child.nodePathLength + 1

        If (node.nodePathLength > Best.nodePathLength)    // If this node is better than Best node
            OR (node.nodePathLength == Best.nodePathLength
            AND node.edgePathLength < Best.edgePathLength):
            Best = node

        Q <— All (node.parent, node) tuples		// Add all nodes with a pointer to descendant child
     return Best

     @return The node which best satisfies the 2 constraints
     */
    public static GraphGenerator.Node getBestNode_2Constraints(GraphGenerator.WeightedGraph graph) {
        GraphGenerator.Node best = null;
        // Queue of (node, child)
        Queue<Tuple<GraphGenerator.Node, GraphGenerator.Node>> queue = new ArrayDeque<>();
        // Add all leaf nodes to queue
        for (GraphGenerator.Node n : graph.getAllLeafNodes()) {
            queue.add(new Tuple<>(n, null));
        }
        while (!queue.isEmpty()) {
            Tuple<GraphGenerator.Node, GraphGenerator.Node> node_child = queue.poll();
            GraphGenerator.Node node = node_child.x;
            GraphGenerator.Node child = node_child.y;
            if (graph.isLeafNode(node)) {
                node.edgePathLength = 0;
                node.nodePathLength = 1;
            }
            else if (isChildBetter_2Constraints(node, child)) {
                node.bestChild = child;
                node.edgePathLength = child.edgePathLength + node.getEdgeValueForChild(child);
                node.nodePathLength = child.nodePathLength + 1;
            }
            // Check if we have discovered a node with a better path
            if (best == null || isRootBetter_2Constraints(node, best)) {
                best = node;
            }
            // Add to the Queue all possible parent paths from node
            for (Tuple<GraphGenerator.Node, Integer> parent: node.parents) {
                queue.add(new Tuple<>(parent.x, node));
            }
        }
        return best;
    }

    /**
     * Determines this (node,child) pair better satisfies the 2 Constraints:
     *      1. The path contains the most amount of query terms.
     *      2. No other path in the G contains an equal number of nodes with a shorter sum of distance weights.
     * @param node Current node in the graph we are working off of
     * @param child Oone of node's children
     */
    private static boolean isChildBetter_2Constraints(GraphGenerator.Node node, GraphGenerator.Node child) {
        // If new path has more query terms OR (path has same number of query terms AND is shorter)
        return node.nodePathLength < child.nodePathLength + 1 ||
               node.nodePathLength == child.nodePathLength &&
               node.edgePathLength > child.edgePathLength + node.getEdgeValueForChild(child);
    }

    /**
     * Determines this root better satisfies the 2 Constraints than the sofar best discovered node:
     *      1. The path contains the most amount of query terms.
     *      2. No other path in the G contains an equal number of nodes with a shorter sum of distance weights.
     * @param root Current node in the graph we are working off of
     * @param best Current best discovered node
     */
    private static boolean isRootBetter_2Constraints(GraphGenerator.Node root, GraphGenerator.Node best) {
        // If new path has more query terms OR (path has same number of query terms AND is shorter)
        return root.nodePathLength > best.nodePathLength ||
               root.nodePathLength == best.nodePathLength &&
               root.edgePathLength < best.edgePathLength;
    }

/************************************ 2Constraints Distance Function ********************************************/

    /**
     * Finds the total score for this graph using the 2Constraints method
     * @param graph A graph representation of a document.
     * @return Score value for the given document represented by the graph
     */
    public static int findDistance_2Constraints(GraphGenerator.WeightedGraph graph) {
        GraphGenerator.Node root = getBestNode_2Constraints(graph);
        return root == null ? 0 : root.nodePathLength << 6 + root.edgePathLength;
    }

/********************************************  Scoring Functions ********************************************/

    public interface ScoringFunction {
        double edgeScore(GraphGenerator.Node n1, GraphGenerator.Node n2, int distance);
        double nodeScore(GraphGenerator.Node n);
    }

    /**
     * EdgeScore uses default term weights and adjustable alpha parameter
     * NodeScore uses default term weight and adjustable beta parameter
     */
    public static class ScoringFunction1 implements ScoringFunction {
        public double alpha = 1;
        public double beta = 1;
        public ScoringFunction1(double alpha) {
            this.alpha = alpha;
        }
        public ScoringFunction1(double alpha, double beta) {
            this.alpha = alpha;
            this.beta = beta;
        }
        public double edgeScore(GraphGenerator.Node n1, GraphGenerator.Node n2, int distance) {
            return (n1.weight + n2.weight)/Math.log(distance + alpha);
        }
        public double nodeScore(GraphGenerator.Node n) {
            return n.weight / beta;
        }
    }

    /**
     * Normalizes the document vector collection and uses the normalized term weights for Edge and Node scores
     * EdgeScore uses tf-idf term weights and adjustable alpha parameter
     * NodeScore uses tf-idf term weight and adjustable beta parameter
     */
    public static class ScoringFunction2 implements ScoringFunction {
        public double alpha = 1;
        public double beta = 1;
        public double gamma = 1;
        public VectorCollection documents;
        public int curDocId;
        public ScoringFunction2(double alpha, VectorCollection documents) {
            this.alpha = alpha;
            this.documents = documents;
        }
        public ScoringFunction2(double alpha, double beta, VectorCollection documents) {
            this.alpha = alpha;
            this.beta = beta;
            this.documents = documents;
        }
        public ScoringFunction2(double alpha, double beta, double gamma, VectorCollection documents) {
            this.alpha = alpha;
            this.beta = beta;
            this.gamma = gamma;
            this.documents = documents;
        }
        public void addCurDocId(int docId) {
            this.curDocId = docId;
        }
        public double edgeScore(GraphGenerator.Node n1, GraphGenerator.Node n2, int distance) {
            DocumentVector doc = (DocumentVector) documents.getVectorById(curDocId);
            double t1 = doc.getNormalizedFrequency(n1.term);
            double t2 = doc.getNormalizedFrequency(n2.term);
            double t3 = gamma * Math.log(distance + alpha);
            return (t1 + t2)/t3;
        }
        public double nodeScore(GraphGenerator.Node n) {
            DocumentVector doc = (DocumentVector) documents.getVectorById(curDocId);
            double t1 = doc.getNormalizedFrequency(n.term);
            return t1 / beta;
        }
    }

    /**
     * Normalizes the document vector collection and uses the normalized term weights for Edge and Node scores
     * EdgeScore uses tf-idf term weights and adjustable alpha parameter
     * NodeScore uses tf-idf term weight and adjustable beta parameter
     */
    public static class ScoringFunction3 implements ScoringFunction {
        public double alpha = 1;
        public double beta = 1;
        public double gamma = 1;
        public VectorCollection documents;
        public int curDocId;
        public ScoringFunction3(double alpha, VectorCollection documents) {
            this.alpha = alpha;
            this.documents = documents;
        }
        public ScoringFunction3(double alpha, double beta, double gamma, VectorCollection documents) {
            this.alpha = alpha;
            this.beta = beta;
            this.gamma = gamma;
            this.documents = documents;
        }
        public void addCurDocId(int docId) {
            this.curDocId = docId;
        }
        public double edgeScore(GraphGenerator.Node n1, GraphGenerator.Node n2, int distance) {
            DocumentVector doc = (DocumentVector) documents.getVectorById(curDocId);
            double t1 = Math.pow(doc.getNormalizedFrequency(n1.term), gamma);
            double t2 = Math.pow(doc.getNormalizedFrequency(n2.term), gamma);
            double t3 = Math.log(distance + alpha);
            return (t1 + t2)/t3;
        }
        public double nodeScore(GraphGenerator.Node n) {
            DocumentVector doc = (DocumentVector) documents.getVectorById(curDocId);
            double t1 = doc.getNormalizedFrequency(n.term);
            return t1 / beta;
        }
    }

/************************************ Scoring Function Best Path Algorithm ********************************************/

    /**
     G(V, E)						    // G with vertices and edges
     Best = NULL                        // The root node containing the best discovered path
     V.score = 0        				// All nodes start out with a score of negative infinity (default)
     Q <— All leaf nodes in AG

     While the Q is not empty:
         (parent, child) = Q.dequeue()

         If parent is a leaf node:
             parent.bestChild = NULL		// End of path (default)

         // If new path has a higher score than the old path
         Else if parent.score < child.score + EdgeScore(parent, child)
             parent.bestChild = child
             parent.score = child.score + Score(parent, child)

         If parent is a root AND			// Node does not have a parent, and if node's score is better than Best's
            parent.score > Best.score):
            Best = parent

         Else:										    // Node has a parent
            Q <— All (parent.parent, parent) tuples		// All parents with a pointer to descendant child
     return Best

     * @param graph
     * @return The node pointed to the highest ranked path
     */
    public static GraphGenerator.Node getBestNode_scoringFunction(GraphGenerator.WeightedGraph graph, ScoringFunction f) {
        GraphGenerator.Node best = null;
        // Queue of (node, child)
        Queue<Tuple<GraphGenerator.Node, GraphGenerator.Node>> queue = new ArrayDeque<>();
        // Add all leaf nodes to queue
        for (GraphGenerator.Node n : graph.getAllLeafNodes()) {
            queue.add(new Tuple<>(n, null));
        }
        while (!queue.isEmpty()) {
            Tuple<GraphGenerator.Node, GraphGenerator.Node> node_child = queue.poll();
            GraphGenerator.Node node = node_child.x;
            GraphGenerator.Node child = node_child.y;

            if (child != null && node.score < child.score + f.edgeScore(node, child, node.getEdgeValueForChild(child))) {
                node.bestChild = child;
                node.score = child.score + f.edgeScore(node, child, node.getEdgeValueForChild(child));
            }
            // Check if we have discovered a node with a better path
            if (best == null || node.score > best.score) {
                best = node;
            }
            // Add to the Queue all possible parent paths from node
            for (Tuple<GraphGenerator.Node, Integer> parent: node.parents) {
                queue.add(new Tuple<>(parent.x, node));
            }
        }
        return best;
    }

/************************************ Scoring Function Distance Functions ********************************************/

    /**
     * Score the document using the longest path in the graph
     * @param graph A graph representation of a document.
     * @param f Scoring function1 with specific values for alpha an beta
     * @return Score value for the given document represented by the graph
     */
    public static double findDistance_function_BestPath(GraphGenerator.WeightedGraph graph, ScoringFunction f) {
        // Compute best path score
        GraphGenerator.Node root = getBestNode_scoringFunction(graph, f);
        return root == null ? 0 : root.score;
    }

    /**
     * Score the document using the longest path in the graph and the total nodes in the graph
     * @param graph A graph representation of a document.
     * @param f Scoring function1 with specific values for alpha an beta
     * @return Score value for the given document represented by the graph
     */
    public static double findDistance_function_BestPath_SumNodes(GraphGenerator.WeightedGraph graph, ScoringFunction f) {
        // Compute best path score
        GraphGenerator.Node root = getBestNode_scoringFunction(graph, f);
        double pathScore = root == null ? 0 : root.score;
        // Compute node score
        double nodeScore = 0;
        for (GraphGenerator.Node node : graph.getAllNodes()) {
            nodeScore += f.nodeScore(node);
        }
        return pathScore + nodeScore;
    }

    /**
     * Score the document based on the total number of edges (connectedness)
     * @param graph A graph representation of a document.
     * @param f Scoring function1 with specific values for alpha an beta
     * @return Score value for the given document represented by the graph
     */
    public static double findDistance_function_SumEdges(GraphGenerator.WeightedGraph graph, ScoringFunction f) {
        // Compute connected score
        double connectedScore = 0;
        // Edge tuples of the form: (x:Node, y:Weight, z:Node)
        for (Tuple3 edge : graph.getAllEdges()) {
            connectedScore += f.edgeScore((GraphGenerator.Node) edge.x, (GraphGenerator.Node) edge.z, (Integer) edge.y);
        }
        return connectedScore;
    }

    /**
     * Score the document based on the total number of edges (connectedness) and the longest path in the graph
     * @param graph A graph representation of a document.
     * @param f Scoring function1 with specific values for alpha an beta
     * @return Score value for the given document represented by the graph
     */
    public static double findDistance_function_SumEdges_BestPath(GraphGenerator.WeightedGraph graph, ScoringFunction f) {
        // Compute connected score
        double connectedScore = 0;
        // Edge tuples of the form: (x:Node, y:Weight, z:Node)
        for (Tuple3 edge : graph.getAllEdges()) {
            connectedScore += f.edgeScore((GraphGenerator.Node) edge.x, (GraphGenerator.Node) edge.z, (Integer) edge.y);
        }
        // Compute best path score
        GraphGenerator.Node root = getBestNode_scoringFunction(graph, f);
        double pathScore = root == null ? 0 : root.score;
        return connectedScore + pathScore;
    }

    /**
     * Score the document based on the total number of edges (connectedness), the longest path in the graph
     * and the total number of nodes in the graph.
     * @param graph A graph representation of a document.
     * @param f Scoring function1 with specific values for alpha an beta
     * @return Score value for the given document represented by the graph
     */
    public static double findDistance_function_SumEdges_BestPath_SumNodes(GraphGenerator.WeightedGraph graph, ScoringFunction f) {
        // Compute connected score
        double connectedScore = 0;
        // Edge tuples of the form: (x:Node, y:Weight, z:Node)
        for (Tuple3 edge : graph.getAllEdges()) {
            connectedScore += f.edgeScore((GraphGenerator.Node) edge.x, (GraphGenerator.Node) edge.z, (Integer) edge.y);
        }
        // Compute best path score
        GraphGenerator.Node root = getBestNode_scoringFunction(graph, f);
        double pathScore = root == null ? 0 : root.score;
        // Compute node score
        double nodeScore = 0;
        for (GraphGenerator.Node node : graph.getAllNodes()) {
            nodeScore += f.nodeScore(node);
        }
        return connectedScore + pathScore + nodeScore;
    }
}
