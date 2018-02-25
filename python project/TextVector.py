# Represents a collection of terms and their frequencies for a vector
# Eric LaBouve (elabouve@calpoly.edu)

from collections import defaultdict


class TextVector:

    def __init__(self):
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
