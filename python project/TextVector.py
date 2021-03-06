# Represents a collection of terms and their frequencies for a vector
# Eric LaBouve (elabouve@calpoly.edu)

from collections import defaultdict


class TextVector:

    def __init__(self):
        # Maps {string term : int frequency}
        self.term_to_freq = defaultdict(int)
        # Maps {string term : float frequency}
        self.normalized_term_to_freq = {}
        # Exact copy of the original text vector
        self.raw_text = ''
        # List of all filtered text vector terms
        self.terms = []
        # List of all filtered text vector terms' part of speech
        self.terms_pos = []
        # ID for this text vector
        self.id = 0
        # The number of terms in this textvector (loaded on request)
        self.length = None

    def __repr__(self):
        s = ''
        s += '<ID=' + str(self.id) + ', {'
        for key, value in self.term_to_freq.items():
            s += str(key) + ':' + str(value) + ', '
        return s + '}>'

    # Returns the number of terms in this textvector
    def __len__(self):
        if self.length is not None:
            return self.length
        term_count = 0
        for term, freq in self.term_to_freq.items():
            term_count += freq
        return term_count

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





















