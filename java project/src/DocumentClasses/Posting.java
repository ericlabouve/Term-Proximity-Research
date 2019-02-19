package DocumentClasses;

import java.util.*;

/**
 * In an inverted index, each term has a corresponding posting that contains the following information:
 *  - Which documents can this term be found in.
 *  - The index of the document this term is found in.
 *
 * @author ericlabouve
 */
public class Posting {
    /** A mapping from a document id to positions inside the indicated document */
    private HashMap<Integer, ArrayList<Integer>> data;

    public Posting() {
        data = new HashMap<Integer, ArrayList<Integer>>();
    }

    /**
     * Obtains a sorted list of document ids for this posting
     * @return A list sorted in increases order of all document ids for this posting
     */
    public ArrayList<Integer> getDocumentIds() {
        ArrayList<Integer> docIds = new ArrayList<>();
        for (Integer a : data.keySet()) {
            docIds.add(a);
        }
        Collections.sort(docIds);
        return docIds;
    }

    /**
     * Obtains the index locations for a term inside the indicated document
     * @param id unique identifier for a document
     * @return the index locations for a term inside the indicated document
     */
    public ArrayList<Integer> getIndexesForDocId(int id) {
        return data.get(id);
    }

    public boolean containsDoc(int id) {
        return data.containsKey(id);
    }

    /**
     * Adds to the posting a document and a corresponding index for a particular term
     * @param docId The document id for the term
     * @param idx The index into the document given by docId for a particular term
     */
    public void add(int docId, int idx) {
        if (!data.containsKey(docId))
            data.put(docId, new ArrayList<Integer>());
        data.get(docId).add(idx);
    }
}
