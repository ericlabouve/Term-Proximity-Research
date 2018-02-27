# Holds distance functions to compute distances between queries and documents
# Eric LaBouve (elabouve@calpoly.edu)

import sys


# doc_or_query is either a DocumentVector or a QueryVector
# x is an integer for the L size for the l-norm
def get_normalized_l_norm(textvector, x):
    running_total = 0
    for term, freq in textvector.normalized_term_to_freq.items():
        running_total += freq ** x
    return running_total ** (1/x)


# Finds the closest document to the query stored in the dist_obj
# This TextVector is intended to be a query
# documents is a VectorCollection
# dist_obj is an object that holds a distance function
# doc_limit - An upper limit for the number of document ids returned per query
def find_closest_docs(documents, dist_obj, doc_limit = sys.maxsize):
    ranked_docs = [] # Holds (doc id, distance)
    for docid, docvector in documents.id_to_textvector.items():
        dist_obj.set_doc(docvector)
        dist = dist_obj.execute()
        ranked_docs.append((docid, dist))
    ranked_docs.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranked_docs][0:doc_limit]


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
    # <d•q> = dot product between d and q
    # ||x|| = sqrt(sum from i=1 to V: Wij^2)
    def execute(self):
        numerator = 0
        # Loop through each word in |query| that maps to a word in |documents| and sum the product of their weights
        for term, weight in self.query.normalized_term_to_freq.items():
            numerator += self.doc.normalized_term_to_freq[term] * weight if term in self.doc.normalized_term_to_freq else 0
        denominator = get_normalized_l_norm(self.query, 2) * get_normalized_l_norm(self.doc, 2)
        return numerator / denominator if denominator != 0 else 0
