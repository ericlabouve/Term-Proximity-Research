package labs;

import DocumentClasses.Posting;
import DocumentClasses.VectorCollection;
import DocumentClasses.TextVector;

import java.util.Collection;

/**
 * Created by Eric on 9/14/17.
 *
 * @author ericlabouve
 */
public class Lab1 {

    /**
     * At the end of the main method, serialize the VectorCollection object to a file.
     * For testing purposes, please print on the screen the word with the highest single
     * document frequency and the frequency, the sum of the distinct number of words in
     * each document over all documents, and the sum of the frequencies of all non-noise
     * words that are stored.
     *
     * Expected output:
     * Word = is
     * Frequency = 33
     * Distinct Number of Words = 90535
     * Total word count = 132267
     */
    public static void main(String[] args) {
        VectorCollection docs = new VectorCollection("/Users/Eric/Desktop/Thesis/programs/src/labs/documents.txt",
                        VectorCollection.VectorType.DOCUMENTS,
                        false, 2);
        Collection<TextVector> tVectors = docs.getTextVectors();

        getStatistics(tVectors);
    }

    /**
     * Prints out lab statistics for lab1
     * @param tVectors
     */
    private static void getStatistics(Collection<TextVector> tVectors) {
        String mostFrequentWord = "";
        int mostFreq = 0;
        int distinctNumWords = 0;
        int totalNumWords = 0;
        for (TextVector v : tVectors) {
            String word = v.getMostFrequentWord();
            int freq = v.getHighestRawFrequency();
            if (freq > mostFreq) {
                mostFrequentWord = word;
                mostFreq = freq;
            }
            distinctNumWords += v.getDistinctWordCount();
            totalNumWords += v.getTotalWordCount();
        }

        // the word with the highest single document frequency and the frequency
        System.out.println("Word = " + mostFrequentWord);
        System.out.println("Frequency = " + mostFreq);
        // the sum of the distinct number of words in each document over all documents
        System.out.println("Distinct Number of Words = " + distinctNumWords);
        // the sum of the frequencies of all non-noise words that are stored
        System.out.println("Total word count = " + totalNumWords);
    }
}
