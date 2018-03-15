# Contains a collection of vectors (Documents or Queries) and preprocess operations
# Eric LaBouve (elabouve@calpoly.edu)

from enum import Enum
from DocumentVector import DocumentVector
from QueryVector import QueryVector
from Posting import Posting
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
import DistanceFunctions as dist_fs
import re, sys, nltk


class VectorType(Enum):
    DOCUMENTS = 1
    QUERIES = 2


class VectorCollection:
    # Set of stop words
    #stop_words = set(stopwords.words('english'))

    stop_words = {"a", "about", "above", "all", "along",
            "also", "although", "am", "an", "and", "any", "are", "aren't", "as", "at",
            "be", "because", "been", "but", "by", "can", "cannot", "could", "couldn't",
            "did", "didn't", "do", "does", "doesn't", "e.g.", "either", "etc", "etc.",
            "even", "ever", "enough", "for", "from", "further", "get", "gets", "got", "had", "have",
            "hardly", "has", "hasn't", "having", "he", "hence", "her", "here",
            "hereby", "herein", "hereof", "hereon", "hereto", "herewith", "him",
            "his", "how", "however", "i", "i.e.", "if", "in", "into", "it", "it's", "its",
            "me", "more", "most", "mr", "my", "near", "nor", "now", "no", "not", "or", "on", "of", "onto",
            "other", "our", "out", "over", "really", "said", "same", "she",
            "should", "shouldn't", "since", "so", "some", "such",
            "than", "that", "the", "their", "them", "then", "there", "thereby",
            "therefore", "therefrom", "therein", "thereof", "thereon", "thereto",
            "therewith", "these", "they", "this", "those", "through", "thus", "to",
            "too", "under", "until", "unto", "upon", "us", "very", "was", "wasn't",
            "we", "were", "what", "when", "where", "whereby", "wherein", "whether",
            "which", "while", "who", "whom", "whose", "why", "with", "without",
            "would", "you", "your", "yours", "yes"}

    def __init__(self, file_path, vector_type, stop_words_on=False, stemming_on=False, min_word_len=2):
        # Maps {doc id : TextVector}
        self.id_to_textvector = {}
        # Maps {string term : {doc id : Posting}}
        self.term_to_postings = defaultdict(dict)
        # Initialize Stemmer
        self.stemmer = PorterStemmer()
        # Parse Documents or Queries
        if vector_type == VectorType.DOCUMENTS:
            self.parse_documents(file_path, stop_words_on, stemming_on, min_word_len)
        elif vector_type == VectorType.QUERIES:
            self.parse_queries(file_path, stop_words_on, stemming_on, min_word_len)

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

# __________________Read VectorCollection File Methods__________________

    # term_offset - absolute position of term inside the document
    # paragraph - paragraph number of term inside the document
    # sentence - sentence number of term inside the document
    # word - word number of term inside the document
    def add_to_inverted_index(self, doc_id, term, term_offset, sentence):
        term_postings = self.term_to_postings[term] # {doc id : Posting}
        if term_postings.get(doc_id) is None:
            p = Posting()
            p.add_doc_id(doc_id)
            term_postings[doc_id] = p
        term_postings[doc_id].add_offset(term_offset)
        term_postings[doc_id].add_sentence(sentence)

    # Fill out inverted index and term frequency table
    # Called from inside parse_documents and parse_queries
    def evaluate_vectors(self, min_word_len: int, stemming_on: bool, stop_words_on: bool):
        # For each text vector
        for textvector_id, textvector in self.id_to_textvector.items():
            cur_offset = 0  # Offset of the term inside the text vector
            sentence_num = 0
            next_sentence = False
            try:
                split_text = re.split("[^a-zA-Z.]+", self.id_to_textvector[textvector_id].raw_text)
                if split_text[-1] == '':
                    del split_text[-1]
                tagged_text = nltk.pos_tag(split_text)
                # Loop through each vector term
    #            for term in re.split("[^a-zA-Z.]+", self.id_to_textvector[textvector_id].raw_text):
                for term, pos in tagged_text:
                    if '.' in term:
                        next_sentence = True
                    term = re.sub(r'[^\w\s]', '', term.lower())  # Remove punctuation
                    if len(term) >= min_word_len:  # Satisfies min length
                        if stemming_on:
                            term = self.stemmer.stem(term)
                        # If (stop words not on and term is not a stop word) or (stop words on)
                        if (not stop_words_on and term not in self.stop_words) or stop_words_on:
                            textvector.add_term(term)  # Add term to term_to_freq
                            textvector.terms.append(term)
                            textvector.terms_pos.append(pos)
                            # Add term to inverted index
                            self.add_to_inverted_index(textvector_id, term, cur_offset, sentence_num)
                            cur_offset += 1
                    if next_sentence:
                        next_sentence = False
                        sentence_num += 1
            except IndexError:
                pass

    def parse_documents(self, file_path: str, stop_words_on: bool, stemming_on: bool, min_word_len: int):
        # Read vectors into memory
        with open(file_path) as file:
            cur_doc_id = -1  # Current Doc Id
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
                elif inside_W:  # In the body of the Document
                    self.id_to_textvector[cur_doc_id].raw_text += re.sub('\n', ' ', line)

        # Fill out inverted index and term frequency table
        self.evaluate_vectors(min_word_len, stemming_on, stop_words_on)

    def parse_queries(self, file_path: str, stop_words_on: bool, stemming_on: bool, min_word_len: int):
        # Read vectors into memory
        with open(file_path) as file:
            cur_query_id = 0  # Current Query Id
            inside_W = False
            for line in file:
                if '.I' in line:  # Contains Id
                    cur_query_id += 1
                    textvector = QueryVector()
                    textvector.add_id(cur_query_id)
                    self.id_to_textvector[cur_query_id] = textvector
                    inside_W = False
                elif '.W' in line:
                    inside_W = True
                elif inside_W:  # In the body of the Document
                    self.id_to_textvector[cur_query_id].raw_text += re.sub('\n', ' ', line)

        # Fill out inverted index and term frequency table
        self.evaluate_vectors(min_word_len, stemming_on, stop_words_on)

# __________________Normalization Methods__________________

    # Normalize the current VectorCollection against the vectors in the given VectorCollection
    # vector_collection is intended to be the document textvector collection
    def normalize(self, vector_collection):
        for id, textvector in self.id_to_textvector.items():
            textvector.normalize(vector_collection)

    # Returns the number of vectors in the collection
    def get_num_vectors(self):
        return len(self.id_to_textvector)

# __________________Distance Methods__________________

    # Computes ALL distances for ALL Queries x Documents.
    # Stores results in a map {Query id : [Doc Ids]} where Doc Ids are
    # sorted in order of relevance.
    # self - Intended to be the Query VectorCollection
    # documents - Intended to be the Documents VectorCollection
    # dist_obj - A object that holds a distance function
    # doc_limit - An upper limit for the number of document ids returned per query
    # query_limit - An upper limit for the number of queries to process
    # returns a map {Query id : [Doc Ids]}
    def find_closest_docs(self, documents, dist_obj, doc_limit=20, query_limit=20) -> map:
        results = {}
        computed = 0
        for qry_id, qry_vector in self.id_to_textvector.items():
            dist_obj.set_query(qry_vector)
            ranked_docs = dist_fs.find_closest_docs(documents, dist_obj, doc_limit=doc_limit)
            results[qry_id] = ranked_docs
            computed += 1
            sys.stdout.write("Q:" + str(qry_id) + " ")
            sys.stdout.flush()
            if computed == query_limit:
                print()
                break
        return results

# __________________Inverted Index Methods__________________

    # Returns the number of vectors that contains the term
    def get_doc_freq(self, term):
        if term in self.term_to_postings:
            return len(self.term_to_postings[term])
        return 0

    # Returns a Posting for a specified term and document
    def get_term_posting_for_doc(self, term: str, doc_id: int) -> Posting:
        # Check if term exists in inverted index
        if term in self.term_to_postings:
            doc_postings = self.term_to_postings[term]
            # Check if document contains term
            if doc_id in doc_postings:
                return doc_postings[doc_id]
        return None


    # def parse_queries(self, file_path: str, stop_words_on: bool, stemming_on: bool, min_word_len: int):
    #     with open(file_path) as file:
    #         cur_query_id = 0 # Current Query Id
    #         cur_query_idx = 0 # Current Query Idx
    #         sentence_num = 0
    #         next_sentence = False
    #         inside_W = False
    #         for line in file:
    #             if '.I' in line: # Contains Id
    #                 cur_query_id += 1
    #                 textvector = QueryVector()
    #                 textvector.add_id(cur_query_id)
    #                 self.id_to_textvector[cur_query_id] = textvector
    #                 inside_W = False
    #             elif '.W' in line:
    #                 inside_W = True
    #                 # Reset for next document
    #                 cur_query_idx = 0
    #                 sentence_num = 0
    #             elif inside_W:  # In the body of the Document
    #                 self.id_to_textvector[cur_query_id].raw_text += re.sub('\n', ' ', line)
    #                 for term in re.split("[^a-zA-Z.]+", line):
    #                     if '.' in term:
    #                         next_sentence = True
    #                     term = re.sub(r'[^\w\s]', '', term.lower())  # Remove punctuation
    #                     if len(term) >= min_word_len: # Satisfies min length
    #                         if stemming_on:
    #                             term = self.stemmer.stem(term)
    #                         # If (stop words not on and term is not a stop word) or (stop words on)
    #                         if (not stop_words_on and term not in self.stop_words) or stop_words_on:
    #                             self.id_to_textvector[cur_query_id].add_term(term)  # Add term to term_to_freq
    #                             self.id_to_textvector[cur_query_id].terms.append(term)
    #                             # Add term to inverted index
    #                             self.add_to_inverted_index(cur_query_id, term, cur_query_idx, sentence_num)
    #                             cur_query_idx += 1
    #                     if next_sentence:
    #                         next_sentence = False
    #                         sentence_num += 1

    # def parse_documents(self, file_path: str, stop_words_on: bool, stemming_on: bool, min_word_len: int):
    #     with open(file_path) as file:
    #         cur_doc_id = -1 # Current Doc Id
    #         cur_doc_idx = 0 # Current Doc Idx
    #         sentence_num = 0
    #         next_sentence = False
    #         inside_W = False
    #         for line in file:
    #             if '.I' in line: # Contains Id
    #                 cur_doc_id = int(re.sub(r'\s+', '', re.sub(r".I", '', line)))  # Remove .I and spaces
    #                 textvector = DocumentVector()
    #                 textvector.add_id(cur_doc_id)
    #                 self.id_to_textvector[cur_doc_id] = textvector
    #                 inside_W = False
    #             elif '.W' in line:
    #                 inside_W = True
    #                 # Reset for next document
    #                 cur_doc_idx = 0
    #                 sentence_num = 0
    #             elif inside_W:  # In the body of the Document
    #                 self.id_to_textvector[cur_doc_id].raw_text += re.sub('\n', ' ', line)
    #                 for term in re.split("[^a-zA-Z.]+", line):
    #                     # .'s indicate end of sentence unless it is surrounded by numbers
    #                     if '.' in term:
    #                         next_sentence = True
    #                     term = re.sub(r'[^\w\s]', '', term.lower())  # Remove punctuation
    #                     if len(term) >= min_word_len:  # Satisfies min length
    #                         if stemming_on:
    #                             term = self.stemmer.stem(term)
    #                         # If (stop words not on and term is not a stop word) or (stop words on)
    #                         if (not stop_words_on and term not in self.stop_words) or stop_words_on:
    #                             self.id_to_textvector[cur_doc_id].add_term(term) # Add term to term_to_freq
    #                             # Add term to inverted index
    #                             self.add_to_inverted_index(cur_doc_id, term, cur_doc_idx, sentence_num)
    #                             cur_doc_idx += 1
    #                     if next_sentence:
    #                         next_sentence = False
    #                         sentence_num += 1


