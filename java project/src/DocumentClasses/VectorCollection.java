package DocumentClasses;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.*;

/**
 * Holds a mapping from IDs to TextVectors
 *
 * @author ericlabouve
 */
public class VectorCollection implements Serializable {

    /** Maps each document ID to a TextVector*/
    private HashMap<Integer, TextVector> idToTextVectorMap;

    /** Inverted Index - A mapping from terms to postings
     * Stores which terms are stored in which documents.
     * Also stores where in the document the term is stored. */
    private HashMap<String, Posting> termToPostingMap;

    /** Whether or not to skip over noise words. */
    private boolean noiseWordsOn;

    /** The minumum character count for a word to be included in the text vector */
    private int minWordLength;

    /** Set of all noise words represented in noiseWordArray */
    private Set<String> noiseWordsSet = null;
    private static  String noiseWordArray[] = {"a", "about", "above", "all", "along",
            "also", "although", "am", "an", "and", "any", "are", "aren't", "as", "at",
            "be", "because", "been", "but", "by", "can", "cannot", "could", "couldn't",
            "did", "didn't", "do", "does", "doesn't", "e.g.", "either", "etc", "etc.",
            "even", "ever", "enough", "for", "from", "further", "get", "gets", "got", "had", "have",
            "hardly", "has", "hasn't", "having", "he", "hence", "her", "here",
            "hereby", "herein", "hereof", "hereon", "hereto", "herewith", "him",
            "his", "how", "however", "i", "i.e.", "if", "in", "into", "it", "it's", "its",
            "me", "more", "most", "mr", "my", "near", "nor", "now", "no", "not", "or", "on", "of", "onto",
            "other", "our", "out", "over", "really", "said", "same", "she",
            "should", "shouldn't", "since", "so", "some", "such",
            "than", "that", "the", "their", "them", "then", "there", "thereby",
            "therefore", "therefrom", "therein", "thereof", "thereon", "thereto",
            "therewith", "these", "they", "this", "those", "through", "thus", "to",
            "too", "under", "until", "unto", "upon", "us", "very", "was", "wasn't",
            "we", "were", "what", "when", "where", "whereby", "wherein", "whether",
            "which", "while", "who", "whom", "whose", "why", "with", "without",
            "would", "you", "your", "yours", "yes"};

    public enum VectorType {
        DOCUMENTS, QUIRIES
    }

    /**
     * Reads the file that is specified as input to populate |idToTextVectorMap|.
     * @param filePath path to an input vectors file.
     * @param vectorType If equal to DOCUMENTS, adds DocumentVector objects to HashMap.
     * @param noiseWordsOn Whether or not to skip over noise words.
     * @param minWordLength The minumum character count for a word to be included in the text vector
     */
    public VectorCollection(String filePath, VectorType vectorType, boolean noiseWordsOn, int minWordLength) {
        this.noiseWordsOn = noiseWordsOn;
        this.minWordLength = minWordLength;
        idToTextVectorMap = new HashMap<Integer, TextVector>();
        termToPostingMap = new HashMap<String, Posting>();
        if (vectorType == VectorType.DOCUMENTS) {
            readDocumentVectors(filePath, noiseWordsOn, minWordLength);
        } else if (vectorType == VectorType.QUIRIES) {
            readQueryVectors(filePath, noiseWordsOn, minWordLength);
        }
    }

    /**
     * Parses the text indicated by |filePath| and populates the idToTextVectorMap with
     * IDs that map to document vectors and the termToPostingMap which is an inverted index
     * that maps terms to documentIds+locations
     *
     * Tag Descriptions:
     * .I - ID
     * .T - Title
     * .A - Author(s)
     * .B - Publisher
     * .W - Text body
     * @param filePath path to a Document vectors file.
     */
    private void readDocumentVectors(String filePath, boolean noiseWordsOn, int minWordLength) {
        try(BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            int curDocId = -1; // Current Document ID
            int curDocIdx = 0; // Current Document Index
            boolean insideW = false;
            String line = br.readLine();
            // Continue until end of file is reached
            while (line != null) {
                // Check if we have reached a new document
                if (line.contains(".I")) {
                    // The current Document ID is the number after .I
                    curDocId = Integer.valueOf(line.replace(".I", "").replaceAll("\\s+",""));
                    TextVector tVector = new DocumentVector();
                    idToTextVectorMap.put(curDocId, tVector);
                    tVector.setId(curDocId);
                    insideW = false;
                }
                // If next line will indicate that we are parsing the text body
                else if (line.contains(".W")) {
                    insideW = true;
                    curDocIdx = 0; // Reset for next document
                }
                // If we are about to parse the text body
                else if (insideW) {
                    // Split into individual words
                    String[] tokens = line.split("[^a-zA-Z]+");
                    for (int i = 0; i < tokens.length; i++) {
                        String term = tokens[i].toLowerCase();
                        // Store words that are appropriate length
                        if (term.length() >= minWordLength) {

                            // If noise words are off and the current word is not a noise word
                            if(!noiseWordsOn && !isNoiseWord(term)) {
                                // Add to text vector
                                getVectorById(curDocId).add(term);
                                // Add to inverted index
                                addTermToPostingMap(term, curDocId, curDocIdx);
                                curDocIdx++;
                            }

                            // Add any word if noise words are on
                            else if (noiseWordsOn) {
                                // Add to text vector
                                getVectorById(curDocId).add(term);
                                // Add to inverted index
                                addTermToPostingMap(term, curDocId, curDocIdx);
                                curDocIdx++;
                            }
                        }
                    }
                }
                line = br.readLine();
            }
        } catch (java.io.FileNotFoundException ex) {
            ex.printStackTrace();
        } catch (java.io.IOException ex) {
            ex.printStackTrace();
        }
    }

    /**
     * Parses the text indicated by |filePath| and populates idToTextVectorMap with
     * IDs that map to document vectors
     *
     * Tag Descriptions:
     * .I - ID
     * .W - Text body
     * @param filePath Path to the Query vector file
     */
    public void readQueryVectors(String filePath, boolean noiseWordsOn, int minWordLength) {
        try(BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            int curQuId = 0; //  Current Query ID
            int curQuIdx = 0; // Current Document Index
            boolean insideW = false;
            String line = br.readLine();
            // Continue until end of file is reached
            while (line != null) {
                // Check if we have reached a new document
                if (line.contains(".I")) {
                    // The current Document is the number after .I
                    curQuId++;
                    TextVector tVector = new QueryVector();
                    idToTextVectorMap.put(curQuId, tVector);
                    tVector.setId(curQuId);
                    insideW = false;
                }
                else if (line.contains(".W")) {
                    insideW = true;
                    curQuIdx = 0; // Reset for next query
                }
                else if (insideW) {
                    String[] tokens = line.split("[^a-zA-Z]+");
                    for (int i = 0; i < tokens.length; i++) {
                        String term = tokens[i].toLowerCase();
                        // Store words that are appropriate length
                        if (term.length() >= minWordLength) {
                            // If noise words are off and the current word is not a noise word
                            if(!noiseWordsOn && !isNoiseWord(tokens[i])) {
                                // Add to text vector
                                getVectorById(curQuId).add(term);
                                // Add to query term to rawText
                                getVectorById(curQuId).appendToRawText(term);
                                // Add to inverted index
                                addTermToPostingMap(term, curQuId, curQuIdx);
                                curQuIdx++;
                            }
                            // Add any word if noise words are on
                            else if (noiseWordsOn) {
                                // Add to text vector
                                getVectorById(curQuId).add(term);
                                // Add to query term to rawText
                                getVectorById(curQuId).appendToRawText(term);
                                // Add to inverted index
                                addTermToPostingMap(term, curQuId, curQuIdx);
                                curQuIdx++;
                            }
                        }
                    }
                }
                line = br.readLine();
            }
        } catch (java.io.FileNotFoundException ex) {
            ex.printStackTrace();
        } catch (java.io.IOException ex) {
            ex.printStackTrace();
        }
    }

/****************************************** Inverted Index Methods ******************************************/

    /**
     * Adds a term to the termToPostingMap with a particular document id and index
     * @param term The word term to insert into the termToPostingMap
     * @param docId The document id for the term
     * @param idx The index into the document given by docId for a particular term
     */
    private void addTermToPostingMap(String term, int docId, int idx) {
        if (!termToPostingMap.containsKey(term))
            termToPostingMap.put(term, new Posting());
        termToPostingMap.get(term).add(docId, idx);
    }

    public Posting getPosting(String term) {
        return termToPostingMap.get(term.toLowerCase());
    }

    public boolean hasPosting(String term) {
        return termToPostingMap.get(term.toLowerCase()) != null;
    }

/****************************************** Other Methods ******************************************/

    public boolean areNoiseWordsOn() {
        return noiseWordsOn;
    }

    public int getMinWordLength() {
        return minWordLength;
    }

    /**
     * Normalizes all the idToTextVectorMap in this collection of idToTextVectorMap with
     * w = tf * idf
     * calls the normalize() method on each document in the collection
     *
     * When you normalize, you pass in the idToTextVectorMap because you care about how many DOCUMENTS contain our search word
     */
    public void normalize(VectorCollection documents) {
        for (TextVector vector : getTextVectors()) {
            vector.normalize(documents);
        }
    }

    /**
     * returns the TextVector for the document with the ID that is given.
     */
    public TextVector getVectorById(int id) {
        return idToTextVectorMap.get(id);
    }

    /**
     * @return the average length of a vector.
     */
    public double getAverageVectorLength() {
        int runningTotal = 0;
        for (TextVector tVector : idToTextVectorMap.values()) {
            runningTotal += tVector.getTotalWordCount();
        }
        return runningTotal / (double) idToTextVectorMap.size();
    }

    /**
     * @return number of idToTextVectorMap
     */
    public int getSize() {
        return idToTextVectorMap.size();
    }

    /**
     * @return a Collection of TextVectors
     */
    public Collection<TextVector> getTextVectors() {
        return idToTextVectorMap.values();
    }

    /**
     * @return a mapping of document id to Text Vector, that is an object of type Set<Map.Entry<Integer, TextVector>>.
     */
    public Set<Map.Entry<Integer, TextVector>> getEntrySet() {
        return idToTextVectorMap.entrySet();
    }

    /**
     * Computes the document frequency for an input word
     * @return the number of vectors in the idToTextVectorMap collection that contains the input word
     */
    public int getDocumentFrequency(String inputWord) {
        int freq = 0;
        for (TextVector doc : idToTextVectorMap.values()) {
            if (doc.contains(inputWord)) {
                freq++;
            }
        }
        return freq;
    }

    /**
     * is the input a noise word.
     */
    private boolean isNoiseWord(String word) {
        // Lazy loading
        if (noiseWordsSet == null) {
            noiseWordsSet = new HashSet<String>(Arrays.asList(VectorCollection.noiseWordArray));
        }
        return noiseWordsSet.contains(word);
    }
}
