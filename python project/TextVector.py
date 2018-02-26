# Represents a collection of terms and their frequencies for a vector
# Eric LaBouve (elabouve@calpoly.edu)

from collections import defaultdict


class TextVector:

    def __init__(self):
        # Maps {string term : int frequency}
        self.term_to_freq = defaultdict(int)
        self.raw_text = ''
        self.id = 0
        pass

    def __repr__(self):
        s = ''
        s += '<ID=' + str(self.id) + ', {'
        for key, value in self.term_to_freq.items():
            s += str(key) + ':' + str(value) + ', '
        return s + '}>'

    def add_id(self, id):
        self.id = id

    def add_term(self, term):
        self.term_to_freq[term] += 1

# __________________Normalization Methods__________________

    # Find the highest term frequency for this textvector
    def get_highest_raw_freq(self):
        highest = 0
        for term, freq in self.term_to_freq.items():
            if freq > highest:
                highest = freq
        return highest

# __________________Distance Methods__________________

    # Finds the closest document to this TextVector
    # This TextVector is inteded to be a query
    # documents is a VectorCollection
    # dist_f is an object that holds a distance function
    def find_closest_docs(self, documents, dist_obj):
        ranked_docs = [] # Holds (doc id, distance)
        for docid, docvector in documents.id_to_textvector.items():
            dist_obj.set_doc(docvector)
            dist = dist_obj.execute()
            ranked_docs.append((docid, dist))
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return ranked_docs





















