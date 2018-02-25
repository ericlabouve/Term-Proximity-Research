package DocumentClasses;

/**
 * Created by Eric on 9/19/17.
 *
 * @author ericlabouve
 */
public interface DocumentDistance {

    /**
     * wWill return the distance between the query and document
     * @param query The search query
     * @param document The document to find the distance from the query
     * @param documents The entire document collection
     * @return The distance from the query to the document
     */
    double findDistance(TextVector query, TextVector document, VectorCollection documents);
}
