package DocumentClasses;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;
import java.util.TreeSet;

/**
 * FIRST ATTEMPT. DO NOT RUN
 *
 * An algorithm for calculating the distance between a query and a document
 * Does not work on queries with less than 2 words
 *
 * @author ericlabouve
 */
public class TermProxDistance1 implements DocumentDistance {

    /** Minimum percentage of terms from the query that must appear in the document*/
    private double threshold;

    public TermProxDistance1 (double threshold) {
        this.threshold = threshold;
    }

    /**
     * Q = Query
     * D = Document collection
     * Inv = Inverted Index of the Document collection
     * y = Threshold for step 5
     *
     * termProximityRanking(Q, Inv, y):
     * 1. d_q = 0, Set the score of q applied to d to 0
     * 2. If y% of elements in q appear in order in document d (using Inv), Else max penalty is given, aka 0
     *      3. bestSubstring <— Shortest substring that contains y% of elements in q in document d
     *      4. For each adjacent pair of words in q, w1 and w2
     *          5. d_q += Distance(w1, w2) in bestSubstring where Distance() is nonlinear, say 2/(e^x + 1)
     *
     * @param query The search query
     * @param document The document to find the distance from the query
     * @param documents The entire document collection, includes the inverted index
     * @return The distance between the query and the document
     */
    @Override
    public double findDistance(TextVector query, TextVector document, VectorCollection documents) {
        assert(query.getRawText().size() >= 2);

        double d_q = 0;
        int minNumTerms = (int) (query.getRawText().size() * threshold); // truncate


        ArrayList<Integer> bestIdxs =
                getShortestString(query.getRawText(), documents, document, minNumTerms);

        System.out.println(bestIdxs);

        if (bestIdxs.size() > 0) {

        }
        return d_q;
    }

    private class Term_Locs {
        public String term;
        public ArrayList<Integer> locs;
        public Term_Locs(String term, ArrayList<Integer> locs) {
            this.term = term;
            this.locs = locs;
        }
    }

    /***********************************************************************************************************
        Determines if |document| contains the minimum number of matching terms in |terms| to be
        considered a relevant document. If so, getShortestString returns the shortest substring
        containing the words
     */
    public ArrayList<Integer> getShortestString(ArrayList<String> qterms,
                                                VectorCollection inv,
                                                TextVector document,
                                                int minNumTerms) {
        // Get the intersect between the
        ArrayList<Term_Locs> termLocs = intersect(qterms, inv, document, minNumTerms);
        if (termLocs == null) return null;

        return findBestSubstringGlobal(minNumTerms, termLocs);
        //return findBestSubstringLocal(minNumTerms, termLocs);
    }

    /**
     * Return the intersect of the terms shared by this query and this document
     * @param qterms The Query Terms
     * @param inv The inverted document with postings
     * @param document The document to intersect with
     * @param minNumTerms The minimum number of intersect words required
     * @return A list of index locations of query words found inside the document
     */
    private ArrayList<Term_Locs> intersect(ArrayList<String> qterms, VectorCollection inv, TextVector document, int minNumTerms) {
        int totalTerms = qterms.size();
        int threshold = totalTerms - minNumTerms; // Number of missing terms allowed
        int termsNotFound = 0;
        int docId = document.getId();

        // List of (term, [locations]) inside our document
        ArrayList<Term_Locs> termLocs = new ArrayList<Term_Locs>();
        for (int i = 0; i < totalTerms; i++) {
            Posting posting = inv.getPosting(qterms.get(i));
            if (posting.containsDoc(docId)) {
                termLocs.add(new Term_Locs(qterms.get(i), posting.getIndexesForDocId(docId)));
            } else {
                termsNotFound++;
            }
        }
        // This document doesn't contain enough matching words
        if (termsNotFound > threshold)
            return null;
        return termLocs;
    }

    /**
     *
     * @param minNumTerms Minimum number of terms to be included in the substring
     * @param termLocs Tuple list holding the terms and their corresponding locations
     * @return
     */
    private ArrayList<Integer> findBestSubstringLocal(int minNumTerms, ArrayList<Term_Locs> termLocs) {
        ArrayList<Integer> best = new ArrayList<>();
        for (int _qT1 = 0; _qT1 < termLocs.size() - 1; _qT1++) { // For each Q term
            Term_Locs qT1 = termLocs.get(_qT1);
            for (int i : qT1.locs) { // For each of Q term 1's locations
                int low = i;
                ArrayList<Integer> A = new ArrayList<>();
                A.add(i); // Add start location
                for (int _qT2 = _qT1 + 1; _qT2 < termLocs.size(); _qT2++) { // For each Q term after Q term 1
                    Term_Locs qT2 = termLocs.get(_qT2);
                    for (int j : qT2.locs) { // For each of Q term 2's locations
                        if (low < j) { // If the loc j comes after low
                            A.add(j);
                            low = j;
                            break;
                        }
                    }
                }
                // If we have included more terms
                if (A.size() > best.size())
                    best = (ArrayList<Integer>) A.clone();
                else if (A.size() == best.size() && strLen(A) < strLen(best))
                    best = (ArrayList<Integer>) A.clone();
            }
        }
        if (best.size() >= minNumTerms)
            return best;
        return null;
    }

    /**
     * Returns the size of the substring indicated by the start and end indexes in arr
     * @param arr Holds the indexes of the terms of Q from our document
     * @return The length of the string indicated by arr
     */
    private int strLen(ArrayList<Integer> arr) {
        if (arr.size() > 0)
            return arr.get(arr.size() - 1) - arr.get(0);
        else
            return -1;
    }

    /***********************************************************************************************************/


    private ArrayList<Integer> findBestSubstringGlobal(int minNumTerms, ArrayList<Term_Locs> termLocs) {

        ArrayList<Integer> best = new ArrayList<>();
        // While the current powerset size is larger/same size as the best substring containing the most Q terms
        for (int curSize = termLocs.size(); curSize >= minNumTerms; curSize--) {
            ArrayList<ArrayList<Term_Pos_Locs>> combs = generatePossibleStrings(termLocs, curSize - 1, curSize - 1);

/*
            for (int _qT1 = 0; _qT1 < termLocs.size() - 1; _qT1++) { // For each Q term
                Term_Locs qT1 = termLocs.get(_qT1);
                for (int i : qT1.locs) { // For each of Q term 1's locations
                    int low = i;
                    ArrayList<Integer> A = new ArrayList<>();
                    A.add(i); // Add start location
                    for (int _qT2 = _qT1 + 1; _qT2 < termLocs.size(); _qT2++) { // For each Q term after Q term 1
                        Term_Locs qT2 = termLocs.get(_qT2);
                        for (int j : qT2.locs) { // For each of Q term 2's locations
                            if (low < j) { // If the loc j comes after low
                                A.add(j);
                                low = j;
                                break;
                            }
                        }
                    }
                    // If we have included more terms
                    if (A.size() > best.size())
                        best = (ArrayList<Integer>) A.clone();
                    else if (A.size() == best.size() && strLen(A) < strLen(best))
                        best = (ArrayList<Integer>) A.clone();
                }
            }
            if (best.size() >= minNumTerms)
                return best;
            return null;
*/
        }
        return null; // temp
    }




    public ArrayList<ArrayList<Term_Pos_Locs>> generatePossibleStrings(ArrayList<Term_Locs> qterms,
                                                                       int minNumTerms,
                                                                       int maxNumTerms) {
        Set<Term_Pos_Locs> set = new TreeSet<>();
        for (int i = 0; i < qterms.size(); i++) {
            set.add(new Term_Pos_Locs(qterms.get(i).term, i, qterms.get(i).locs));
        }
        PowerSet<Term_Pos_Locs> pset = new PowerSet<Term_Pos_Locs>(set, minNumTerms, maxNumTerms);
        ArrayList<ArrayList<Term_Pos_Locs>> posStrings = new ArrayList<>();
        for (Set<Term_Pos_Locs> s : pset) {
            ArrayList<Term_Pos_Locs> arr = new ArrayList<>(s);
            Collections.sort(arr);
            if (!posStrings.contains(arr))
                posStrings.add(arr);
        }
        return posStrings;
    }


    /**
     * Tuple for Term and its position
     */
    public class Term_Pos_Locs implements Comparable<Term_Pos_Locs> {

        public String term; // The term as a string
        public int pos; // The location inside the query
        public ArrayList<Integer> locs; // The locations inside the target document

        public Term_Pos_Locs(String term, int pos, ArrayList<Integer> locs) {
            this.term = term;
            this.pos = pos;
            this.locs = locs;
        }

        @Override
        public int compareTo(Term_Pos_Locs o) {
            if (this.pos < o.pos)
                return -1;
            else if (this.pos > o.pos)
                return 1;
            return 0;
        }

        @Override
        public String toString() {
            return term;
        }
    }

    /***********************************************************************************************************/
}
