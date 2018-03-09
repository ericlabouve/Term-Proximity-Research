# Represents a single Query Vector
# Eric LaBouve (elabouve@calpoly.edu)

from TextVector import TextVector
import math


class QueryVector(TextVector):

    def __init__(self):
        super().__init__()
        self.normalized_term_to_freq = {}
        self.terms = [] # Copy of query terms in order in which they appear

    def __repr__(self):
        return super().__repr__()

    # Normalizes term_to_freq from TextVector superclass according to:
    # w = (0.5 + tf) * idf
    # tf = (0.5 * f) / max{f1, f2, ..., f|V|} where f is the raw frequency count of a term t
    # idf = log2(N/df) where N is the total number of documents in the system and df
    #      is the number of documents in which the term t appears at least once.
    def normalize(self, vector_collection):
        max_tf = self.get_highest_raw_freq()
        # Normalize each of the terms in this document vector
        for term, freq in self.term_to_freq.items():
            tf = (0.5 * freq) / max_tf
            df = vector_collection.get_doc_freq(term)
            if df == 0:
                idf = 0
            else:
                idf = math.log2(vector_collection.get_num_vecotrs() / df)
            w = (0.5 + tf) * idf
            self.normalized_term_to_freq[term] = w
