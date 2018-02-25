# Contains a collection of vectors (Documents or Queries) and preprocess operations
# Eric LaBouve (elabouve@calpoly.edu)

from enum import Enum
from DocumentVector import DocumentVector
from QueryVector import QueryVector
from Posting import Posting
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from string import punctuation
import re

class VectorType(Enum):
    DOCUMENTS = 1
    QUERIES = 2


class VectorCollection:
    # Set of stop words
    stop_words = set(stopwords.words('english'))

    def __init__(self, file_path, stop_words_on, stemming_on, min_word_len, vector_type):
        # Maps TextVector ID to TextVector
        self.id_to_textvector = {}
        # Inverted Index: Maps a term to a dict of {doc id : Posting}
        self.term_to_postings = defaultdict(dict)
        # Initialize Stemmer
        self.stemmer = PorterStemmer()
        # Parse Documents or Queries
        if vector_type == VectorType.DOCUMENTS:
            self.parse_documents(file_path, stop_words_on, stemming_on, min_word_len)
        elif vector_type == VectorType.QUERIES:
            self.parse_documents(file_path, stop_words_on, stemming_on, min_word_len)

    def __repr__(self):
        s = 'id_to_textvector:\n\tKEY\t\tVALUE\n'
        a = 0
        for key, value in self.id_to_textvector.items():
            s += '\t' + str(key) + '\t\t' + str(value) + '\n'
            if a == 5:
                break
            a += 1
        s += 'term_to_postings:\n\tKEY\t\tVALUE\n'
        a = 0
        for key, value in self.term_to_postings.items():
            s += '\t' + str(key) + '\t\t' + str(value) + '\n'
            if a == 5:
                break
            a += 1
        return s

    def add_to_inverted_index(self, doc_id, term, term_offset):
        term_postings = self.term_to_postings[term] # {doc id : Posting}
        if term_postings.get(doc_id) is None:
            p = Posting()
            p.add_doc_id(doc_id)
            term_postings[doc_id] = p
        term_postings[doc_id].add_offset(term_offset)

    def parse_documents(self, file_path, stop_words_on, stemming_on, min_word_len):
        with open(file_path) as file:
            cur_doc_id = -1 # Current Doc Id
            cur_doc_idx = 0 # Current Doc Idx
            inside_W = False
            for line in file:
                if '.I' in line: # Contains Id
                    cur_doc_id = int(re.sub(r'\s+', '', re.sub(r".I", '', line)))  # Remove .I and spaces
                    textvector = DocumentVector()
                    textvector.add_id(cur_doc_id)
                    self.id_to_textvector[cur_doc_id] = textvector
                    inside_W = False
                elif '.W' in line:
                    inside_W = True
                    cur_doc_idx = 0 # Reset for next document
                elif inside_W: # In the body of the Document
                    for term in line.split():
                        term = re.sub(r'[^\w\s]', '', term.lower()) # Remove punctuation
                        if len(term) >= min_word_len: # Satisfies min length
                            if stemming_on:
                                term = self.stemmer.stem(term)
                            # If (stop words not on and term is not a stop word) or (stop words on)
                            if (not stop_words_on and term not in self.stop_words) or stop_words_on:
                                self.id_to_textvector[cur_doc_id].add_term(term) # Add term to term_to_freq
                                # Add term to inverted index
                                self.add_to_inverted_index(cur_doc_id, term, cur_doc_idx)
                                cur_doc_idx += 1

    def parse_queries(self, file_path, stop_words_on, stemming_on, min_word_len):
        with open(file_path) as file:
            cur_doc_id = -1 # Current Doc Id
            cur_doc_idx = 0 # Current Doc Idx
            inside_W = False
            for line in file:
                if '.I' in line: # Contains Id
                    cur_doc_id = int(re.sub(r'\s+', '', re.sub(r".I", '', line)))  # Remove .I and spaces
                    textvector = QueryVector()
                    textvector.add_id(cur_doc_id)
                    self.id_to_textvector[cur_doc_id] = textvector
                    inside_W = False
                elif '.W' in line:
                    inside_W = True
                    cur_doc_idx = 0 # Reset for next document
                elif inside_W: # In the body of the Document
                    for term in line.split():
                        term = re.sub(r'[^\w\s]', '', term.lower()) # Remove punctuation
                        if len(term) >= min_word_len: # Satisfies min length
                            if stemming_on:
                                term = self.stemmer.stem(term)
                            # If (stop words not on and term is not a stop word) or (stop words on)
                            if (not stop_words_on and term not in self.stop_words) or stop_words_on:
                                self.id_to_textvector[cur_doc_id].add_term(term) # Add term to term_to_freq
                                # Add term to inverted index
                                self.add_to_inverted_index(cur_doc_id, term, cur_doc_idx)
                                cur_doc_idx += 1





























