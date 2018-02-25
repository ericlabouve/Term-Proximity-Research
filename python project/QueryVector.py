# Represents a single Query Vector
# Eric LaBouve (elabouve@calpoly.edu)

from TextVector import TextVector


class QueryVector(TextVector):

    def __init__(self):
        super().__init__()
        self.normalized_term_to_freq = {}

    def __repr__(self):
        return super().__repr__()