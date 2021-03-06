# In an inverted index, terms are mapped to Postings.
# For the key term, contains a list of term offsets inside the document with the given id
# For the key term, contains the frequency of this term inside the document with the given id
#
# Eric LaBouve (elabouve@calpoly.edu)

class Posting:

    def __init__(self):
        self.doc_id = 0
        self.freq_count = 0
        self.offsets = []  # Absolute index position inside the document
        self.sentence = []  # Index of which sentence the term appears in

    def __repr__(self):
        s = '<Doc ID=' + str(self.doc_id) + ', Count=' + str(self.freq_count) + ', '
        return s + str(self.offsets) + '>'

    def add_offset(self, offset: int):
        self.freq_count += 1
        self.offsets.append(offset)

    def add_sentence(self, sentence: int):
        self.sentence.append(sentence)

    def add_doc_id(self, doc_id: int):
        self.doc_id = doc_id
