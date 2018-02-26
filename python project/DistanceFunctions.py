# Holds distance functions to compute distances between queries and documents
# Eric LaBouve (elabouve@calpoly.edu)


# doc_or_query is either a DocumentVector or a QueryVector
# x is an integer for the L size for the l-norm
def get_normalized_l_norm(doc_or_query, x):
    running_total = 0
    for term, freq in doc_or_query.normalized_term_to_freq.items():
        running_total += freq ** x
    return running_total ** -x if running_total != 0 else 0


class CosineFunction:
    def __init__(self, vector_collection):
        self.vector_collection = vector_collection
        self.query = None
        self.doc = None

    def set_query(self, query_tv):
        self.query = query_tv

    def set_doc(self, doc_tv):
        self.doc = doc_tv

    # Computes the cosine similarity between a query and a document
    # cosine(d,q) = <d•q>/(||d|| * ||q||)
    # <d•q> = sum from i=1 to V: (Wij * Wiq)
    # ||x|| = sqrt(sum from i=1 to V: Wij^2)
    def execute(self):
        numerator = 0
        # Loop through each word in |query| that maps to a word in |documents| and sum the product of their weights
        for term, weight in self.query.normalized_term_to_freq.items():
            numerator += self.doc.normalized_term_to_freq[term] if term in self.doc.normalized_term_to_freq else 0
        denominator = get_normalized_l_norm(self.query, 2) * get_normalized_l_norm(self.doc, 2)
        return numerator / denominator if denominator != 0 else 0
