package DocumentClasses;

import com.sun.corba.se.impl.orbutil.graph.Graph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 Query:
 A B C D

 Document:
 B B D A C D A B A C
 0 1 2 3 4 5 6 7 8 9

 A	 B	C	D
 3	 0	4	2
 6	 1	9	5
 8	 7

 AG Vertexes:
        A	B	C	D
 0 - 		*
 1 - 		*
 2 - 				*
 3 - 	*
 4 - 			*
 5 - 				*
 6 - 	*
 7 - 		*
 8 - 	*
 9 - 			*

 * @author ericlabouve
 */
public class GraphGenerator {

    // Counter for node id values
    private static int idCounter = 0;

    public class Node {
        // Unique identifier for the node
        public int id;
        // The query term
        public String term;
        // Index of query term in document
        public int index;
        // The significance of this term in our query
        public double weight;
        // Score for getBestNode_scoringFunction
        public double score;
        // List of (parent nodes, edge weights)
        public ArrayList<Tuple<Node, Integer>> parents;
        // List of (child nodes, edge weights)
        public ArrayList<Tuple<Node, Integer>> children;

        // Metadata for GraphReader:
        public int edgePathLength;  // Shortest path based on cumulative edge weights
        public int nodePathLength;  // Shortest path based on cumulative node count
        public Node bestChild;      // Pointer to child which satisfies above 2 constraints

        public Node(String term, int index) {
            this.id = idCounter++;
            this.term = term;
            this.index = index;
            this.weight = 1;
            this.score = 0;
            parents = new ArrayList<>();
            children = new ArrayList<>();

            edgePathLength = Integer.MAX_VALUE;
            nodePathLength = 0;
            bestChild = null;
        }

        public void addParent(Node parent, int weight) {
            parents.add(new Tuple<>(parent, weight));
        }

        public void addChild(Node parent, int weight) {
            children.add(new Tuple<>(parent, weight));
        }

        /**
         * @param parent Parent node that connects to this node
         * @return Edge weight between this node and parent node
         */
        public int getEdgeValueForParent(Node parent) {
            for (Tuple<Node, Integer> pair : parents) {
                if (pair.x.id == parent.id)
                    return pair.y;
            }
            return Integer.MAX_VALUE;
        }

        /**
         * @param child Child node that connects to this node
         * @return Edge weight between this node and child node
         */
        public int getEdgeValueForChild(Node child) {
            for (Tuple<Node, Integer> pair : children) {
                if (pair.x.id == child.id)
                    return pair.y;
            }
            return Integer.MAX_VALUE;
        }

        @Override
        public boolean equals(Object _other) {
            Node other = (Node) _other;
            return this.term.equals(other.term) && this.index == other.index &&
                    this.parents.equals(other.parents) && this.children.equals(other.children);
        }

        @Override
        public String toString() {
            return "(" + term + ":" + index + ")";
        }
    }

    public class WeightedGraph {

        // Mapping from node labels to a list of nodes
        public HashMap<String, ArrayList<Node>> nodeMap;
        // Query terms
        public ArrayList<String> qTerms;

        public WeightedGraph(ArrayList<String> qTerms) {
            nodeMap = new HashMap<>();
            this.qTerms = qTerms;
        }

        public void addNode(Node node) {
            nodeMap.putIfAbsent(node.term, new ArrayList<Node>());
            nodeMap.get(node.term).add(node);
        }

        /**
         * @return All nodes with the specific term
         */
        public ArrayList<Node> getNodes(String term) {
            return nodeMap.get(term);
        }

        /**
         * @return All nodes
         */
        public ArrayList<Node> getAllNodes() {
            ArrayList<Node> allNodes = new ArrayList<>();
            for (ArrayList<Node> nodesForTerm : nodeMap.values()) {
                allNodes.addAll(nodesForTerm);
            }
            return allNodes;
        }

        public boolean isLeafNode(Node node) {
            return node.children.size() == 0;
        }

        public boolean isRootNode(Node node) {
            return node.parents.size() == 0;
        }

        /**
         * @return All nodes which dont have any children
         */
        public ArrayList<Node> getAllLeafNodes() {
            ArrayList<Node> leafNodes = new ArrayList<>();
            for (Node n : getAllNodes()) {
                if (isLeafNode(n)) {
                    leafNodes.add(n);
                }
            }
            return leafNodes;
        }

        /**
         * @return List of all unique edges of the form (Node, Weight, Node)
         */
        public ArrayList<Tuple3<Node, Integer, Node>> getAllEdges() {
            ArrayList<Tuple3<Node, Integer, Node>> edges = new ArrayList<>();
            HashSet<Node> seenSet = new HashSet<>();
            for (Node n : getAllNodes()) {
                ArrayList<Tuple<Node, Integer>> parents_children = new ArrayList<>();
                parents_children.addAll(n.parents);
                parents_children.addAll(n.children);

                for (Tuple<Node, Integer> pair : parents_children) {
                    Tuple3<Node, Integer, Node> edge = new Tuple3<>(n, pair.y, pair.x);
                    if (!seenSet.contains(pair.x))
                        edges.add(edge);
                }
                seenSet.add(n);
            }
            return edges;
        }

        public String toString() {
            int max = 0;
            for (Node n : getAllNodes()) {
                if (n.index > max)
                    max = n.index;
            }
            int rows = max + 2;
            int cols = qTerms.size() + 1;
            String[][] matrix = new String[rows][cols];
            // Fill in top row with query terms
            for (int col = 1; col < cols; col++)
                matrix[0][col] = "Q" + Integer.toString(col);
            // Fill in left column with indexes
            for (int row = 1; row < rows; row++)
                matrix[row][0] = Integer.toString(row - 1);
            // Fill in contents of table
            for (Node n : getAllNodes()) {
                matrix[n.index + 1][qTerms.indexOf(n.term) + 1] = "*";
            }
            StringBuffer str = new StringBuffer();
            str.append("Query:\n");
            int i = 1;
            for (String qTerm : qTerms) {
                str.append(Integer.toString(i) + "-" + qTerm + " ");
                i++;
            }
            str.append("\n\n");
            for (int row = 0; row < rows; row++) {
                boolean hasValue = false;
                // Check if row contains a value
                for (int col = 1; col < cols; col++) {
                    if (matrix[row][col] != null) {
                        hasValue = true;
                        break;
                    }
                }
                if (hasValue) {
                    for (int col = 0; col < cols; col++) {
                        if (matrix[row][col] == null)
                            str.append("\t");
                        else
                            str.append(matrix[row][col] + "\t");
                    }
                    str.append("\n");
                }
            }
            str.append("\n");
            // Print all edges
            str.append("Edges:\n");
            for (Tuple3<Node, Integer, Node> edge : getAllEdges()) {
                String n1 = "Q" + Integer.toString(qTerms.indexOf(edge.x.term) + 1);
                String n2 = "Q" + Integer.toString(qTerms.indexOf(edge.z.term) + 1);
                str.append(n1 + "--" + edge.y + "--" + n2 + "\n");
            }
            return str.toString();
        }
    }

    /**
     Q <— [q0, q1, … qn] 		// Query and the ordered list of terms
     AG(V, E)					// An empty graph of vertices and edges
     Inv						// Inverted Index
     maxDist					// User specified max distance between two terms

     for (i = 0; i <= n; i++) 	    // For each query term in reverse order with i being the index into the Query
        qi = Q[i]				    // Obtain query term
        IdxList = Inv(qi, D)	    // Obtain list of indexes for this query term in this document

        for each index in IdxList, idx:
            newVertex = (qi, idx)		// Contains the query term and the query term’s index

            for each vertex in AG(V, E), v: 		    // Empty the first time through
                if newVertex.name != v.name             // Term names are not the same
                AND newVertex.idx > v.idx: 		        // If our current query term comes after v
                AND newVertex.idx - v.idx ≤ maxDistance // and dist between these terms is small enough
                    edgeWeight = newVertex.idx - v.id
                    AG.add(v, newVertex, edgeWeight)	    // Make an edge between the two term vertices
                                                            // with weight equal to the distance between the two terms
            AG <— newVertex

     * @return A root node connecting all the starting
     */
    public WeightedGraph generateTree(QueryVector query, VectorCollection inv, int docID, int maxDist) {
        ArrayList<String> qTerms = query.getRawText();
        WeightedGraph tree = new WeightedGraph(qTerms);
        // For each term in the query
        for (String qTerm : qTerms) {
            if (inv.hasPosting(qTerm) && inv.getPosting(qTerm).containsDoc(docID)) {
                // Get indexes of the query term in this document
                ArrayList<Integer> idxList = inv.getPosting(qTerm).getIndexesForDocId(docID);
                // For each index
                for (int idx : idxList) {
                    Node thisNode = new Node(qTerm, idx);
                    // For each node in our graph
                    for (Node otherNode : tree.getAllNodes()) {
                        // This q term is not the same term as other q term
                        if (!thisNode.term.equals(otherNode.term)
                            // This q term comes after other q term
                            && thisNode.index > otherNode.index
                            // Distance between this q term and other q term is small
                            && thisNode.index - otherNode.index <= maxDist ) {
                                int edgeWeight = thisNode.index - otherNode.index;
                                thisNode.addParent(otherNode, edgeWeight);
                                otherNode.addChild(thisNode, edgeWeight);
                        }
                    }
                    tree.addNode(thisNode);
                }
            }
        }
        return tree;
    }
}
