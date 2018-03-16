# Holds distance functions to compute distances between queries and documents
# Eric LaBouve (elabouve@calpoly.edu)
#
# DATASET = cran: query_limit=225, doc_limit=20, stemming_on=True
# Unmodified Cosine: MAP=0.25144842545683216
# Unmodified Okapi: MAP=0.25494314903429477
# is_eary=True MAP=0.2657067430393868
# is_noun=True (I=1.1) MAP=0.2553366000643001
# is_verb=True (I=1.1) MAP=0.2536578704799122
# is_adj_noun_pairs=True (I=1.8): MAP=0.25654817087197906
# is_adv_verb_pairs=True (I=1.2): MAP=0.2550557616469074
# is_adj_noun_pairs=True (I=1.8), is_adv_verb_pairs=True (I=1.2): MAP=0.25654817087197906
# is_eary=True, is_adj_noun_pairs=True (I=1.8): MAP=0.26875021721497516
# is_eary=True, is_adj_noun_pairs=True (I=1.8), is_adv_verb_pairs=True (I=1.2): MAP=0.26875021721497516
#
# DATASET = cran: query_limit=225, doc_limit=20, stemming_on=False
# Unmodified Okapi: MAP=0.23963612749870844
# is_eary=True MAP=0.2455933914749088
# is_adj_noun_pairs=True: MAP=0.23666999367828484, Influence=1.8
# is_eary=True and is_adj_noun_pairs=True: MAP=0.242043464756276, Influence=1.8

import math
import sys

import DocumentVector
import QueryVector
import VectorCollection
import WordNet as wn


class DistanceFunction:
    def __init__(self, vector_collection: VectorCollection):
        self.vector_collection = vector_collection
        self.query = None
        self.doc = None

    def set_query(self, query_tv: QueryVector):
        self.query = query_tv

    def set_doc(self, doc_tv: DocumentVector):
        self.doc = doc_tv


# Finds the closest document to the query stored in the dist_obj
# This TextVector is intended to be a query
# documents is a VectorCollection
# dist_obj is an object that holds a distance function
# doc_limit - An upper limit for the number of document ids returned per query
def find_closest_docs(documents: VectorCollection, dist_obj: DistanceFunction, doc_limit=20) -> list:
    ranked_docs = []  # Holds (doc id, distance)
    for docid, docvector in documents.id_to_textvector.items():
        dist_obj.set_doc(docvector)
        dist = dist_obj.execute()
        ranked_docs.append((docid, dist))
    ranked_docs.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranked_docs][0:doc_limit]


# ______________________ Cosine Functions ______________________

# doc_or_query is either a DocumentVector or a QueryVector
# x is an integer for the L size for the l-norm
def get_normalized_l_norm(textvector: VectorCollection, x: int) -> float:
    running_total = 0
    for term, freq in textvector.normalized_term_to_freq.items():
        running_total += freq ** x
    return running_total ** (1 / x)


class CosineFunction(DistanceFunction):
    def __init__(self, vector_collection):
        super().__init__(vector_collection)

    # Computes the cosine similarity between a query and a document
    # cosine(d,q) = <d•q>/(||d|| * ||q||)
    # <d•q> = dot product between d and q
    # ||x|| = sqrt(sum from i=1 to V: Wij^2)
    def execute(self) -> float:
        numerator = 0
        # Loop through each word in |query| that maps to a word in |documents| and sum the product of their weights
        for term, weight in self.query.normalized_term_to_freq.items():
            numerator += self.doc.normalized_term_to_freq[
                             term] * weight if term in self.doc.normalized_term_to_freq else 0
        denominator = get_normalized_l_norm(self.query, 2) * get_normalized_l_norm(self.doc, 2)
        return numerator / denominator if denominator != 0 else 0


# ______________________ Okapi Functions ______________________

# Computes and returns the average document length for vectors in vector_collection
# vector_collection - Intended to be the document collection
def compute_avdl(vector_collection: VectorCollection) -> float:
    total_words = 0
    num_docs = 0
    for id, textvector in vector_collection.id_to_textvector.items():
        total_words += len(textvector)
        num_docs += 1
    return total_words / num_docs


# Positively boosts the Okapi score
# cur_score - Score to be boosted
# influence - Boosts the current score by value of influence
def boost(cur_score: float, influence: float) -> float:
    if cur_score > 0:
        cur_score *= influence
    else:
        cur_score += influence * abs(cur_score) - abs(cur_score)
    return cur_score


# cran: query_limit=225, doc_limit=20: MAP=0.25494314903429477
class OkapiFunction(DistanceFunction):
    def __init__(self, vector_collection: VectorCollection):
        super().__init__(vector_collection)
        self.avdl = compute_avdl(vector_collection)
        self.num_docs = len(vector_collection.id_to_textvector)
        self.k1 = 1.2
        self.k2 = 100
        self.b = 0.75

    # SUM( [ln( (N-dfi+0.5)/(dfi+0.5) )] * [ ((k1+1)*fij)/(k1*(1-b+b*dlj/avdl)+fij) ] * [((k2+1)*fiq)/(k2+fiq)])
    # ti is a term
    # fij is the raw frequency count of term ti in document dj
    # fiq is the raw frequency count of term ti in query q
    # N is the total number of documents in the collection
    # dfi is the number of documents that contain the term ti
    # dlj is the document length (in bytes) of d
    # avdl is the average document length of the collection
    def execute(self) -> float:
        sum = 0
        for term, weight in self.query.term_to_freq.items():
            dfi = self.vector_collection.get_doc_freq(term)
            fij = self.doc.term_to_freq[term]
            dlj = len(self.doc)
            fiq = self.query.term_to_freq[term]

            first_term = math.log((self.num_docs - dfi + 0.5) / (dfi + 0.5))
            second_term = ((self.k1 + 1) * fij) / (self.k1 * (1 - self.b + self.b * dlj / self.avdl) + fij)
            third_term = ((self.k2 + 1) * fiq) / (self.k2 + fiq)

            product = first_term * second_term * third_term
            sum += product
        return sum


class OkapiModFunction(DistanceFunction):
    def __init__(self, vector_collection: VectorCollection,
                 is_early=False,
                 is_close_pairs=False,
                 is_noun=False, noun_influence=1.0,
                 is_verb=False, verb_influence=1.0,
                 is_adj_noun_pairs=False, adj_noun_pairs_influence=1.8,
                 is_adv_verb_pairs=False, adv_verb_pairs_influence=1.2):

        super().__init__(vector_collection)
        # Okapi variables
        self.avdl = compute_avdl(vector_collection)
        self.num_docs = len(vector_collection.id_to_textvector)
        self.k1 = 1.2
        self.k2 = 100
        self.b = 0.75
        self.last_term_pos = None
        # is_early variables
        self.is_early = is_early
        # is_close_pairs variables
        self.is_close_pairs = is_close_pairs
        self.last_term = None
        # is_noun variables
        self.is_noun = is_noun
        self.noun_influence = noun_influence
        # is_verb variables
        self.is_verb = is_verb
        self.verb_influence = verb_influence
        # is_adj_noun_pairs variables
        self.is_adj_noun_pairs = is_adj_noun_pairs
        self.adj_noun_pairs_influence = adj_noun_pairs_influence
        # is_adj_noun_pairs variables
        self.is_adv_verb_pairs = is_adv_verb_pairs
        self.adv_verb_pairs_influence = adv_verb_pairs_influence

    def execute(self) -> float:
        sum = 0
        terms = []
        # Traverse query terms in order of how they appear
        # Do not want to double score the same term
        for term, pos in zip(self.query.terms, self.query.terms_pos):
            product = 0
            if term not in terms:
                dfi = self.vector_collection.get_doc_freq(term)
                fij = self.doc.term_to_freq[term]
                dlj = len(self.doc)
                fiq = self.query.term_to_freq[term]
                first_term = math.log((self.num_docs - dfi + 0.5) / (dfi + 0.5))
                second_term = ((self.k1 + 1) * fij) / (self.k1 * (1 - self.b + self.b * dlj / self.avdl) + fij)
                third_term = ((self.k2 + 1) * fiq) / (self.k2 + fiq)

                product = first_term * second_term * third_term

            if self.is_early:
                product = boost(product, self.early_term(term))

            if self.is_close_pairs:
                product = boost(product, self.close_pairs(term))
                self.last_term = term

            if self.is_noun:
                if wn.is_noun(pos):
                    product = boost(product, self.noun_influence)

            if self.is_verb:
                if wn.is_verb(pos):
                    product = boost(product, self.verb_influence)

            if self.is_adj_noun_pairs:
                # If the last adjective in the query is before the current noun
                if self.adj_noun_pairs(term, pos):
                    product = boost(product, self.adj_noun_pairs_influence)
                self.last_term_pos = (term, pos)

            if self.is_adv_verb_pairs:
                # If the last adverb in the query is before the current verb
                if self.adv_verb_pairs(term, pos):
                    product = boost(product, self.adv_verb_pairs_influence)
                self.last_term_pos = (term, pos)

            terms.append(term)
            sum += product
        return sum

    # Compute as a percentage of the way through the document.
    # Range of values: [const, 1] where default const=2
    # boost = (2 * dl - term_loc) / dl
    def early_term(self, term, const=2):
        dl = 1 if len(self.doc) == 0 else len(self.doc)
        posting = self.vector_collection.get_term_posting_for_doc(term, self.doc.id)
        term_loc = dl if posting is None else posting.offsets[0]
        return (const * dl - term_loc) / dl

    # Does not give good results. Gives too much weight to adjacent common words
    #   ex: 'high' 'speed', which detracts from the main subject of the query
    # MAP=0.23606341931968342
    # Gives an extra boost to adjacent query terms that are near each other
    # in the document. Idea is to give nonlinear reward for terms that are close
    # Evaluate only the closest pair of terms between the two postings.
    # Range of values: [2, 0]
    # boost = 2 / min_distance
    def close_pairs(self, term):
        if self.last_term is None:
            self.last_term = term
            return 1
        posting1 = self.vector_collection.get_term_posting_for_doc(term, self.doc.id)
        posting2 = self.vector_collection.get_term_posting_for_doc(self.last_term, self.doc.id)
        # Determine if both terms appear in the document and have not been stemmed to the same term
        if posting1 is None or posting2 is None or term == self.last_term:
            return 1
        # Find the two closest pairs from the two sorted arrays
        ar1 = posting1.offsets
        ar2 = posting2.offsets
        i = j = 0
        min_dif = sys.maxsize
        while i < len(ar1) and j < len(ar2):
            dif = abs(ar1[i] - ar2[j])
            min_dif = min(min_dif, dif)
            if ar1[i] > ar2[j]:
                j += 1
            else:
                i += 1
        if i < len(ar1):
            j -= 1
            while i < len(ar1):
                dif = abs(ar1[i] - ar2[j])
                min_dif = min(min_dif, dif)
                i += 1
        else:
            i -= 1
            while j < len(ar2):
                dif = abs(ar1[i] - ar2[j])
                min_dif = min(min_dif, dif)
                j += 1
        result = 2 / min_dif
        return result

    # Gives an extra boost to adjectives and nouns that are found near each other,
    # either in the same sentence or the same clause
    def adj_noun_pairs(self, term, pos):
        if self.last_term_pos is None:
            self.last_term_pos = (term, pos)
            return False
        # If there is an adjacent adjective and noun in the query
        if wn.is_adjective(self.last_term_pos[1]) and wn.is_noun(pos):
            # Get term locations inside this document
            posting1 = self.vector_collection.get_term_posting_for_doc(term, self.doc.id)
            posting2 = self.vector_collection.get_term_posting_for_doc(self.last_term_pos[0], self.doc.id)
            # Determine if both terms appear in the document
            if posting1 is None or posting2 is None:
                return False
            # Boost if ADJ and NOUN appear in same sentence and same order as query
            sentences = zip(posting1.sentence, posting2.sentence)
            for idx, s1_s2 in enumerate(sentences):
                s1 = s1_s2[0]
                s2 = s1_s2[1]
                if s1 == s2 and posting1.offsets[idx] > posting2.offsets[idx]:
                    return True
        return False

    # Gives an extra boost to adverbs and verbs that are found near each other,
    # either in the same sentence or the same clause
    def adv_verb_pairs(self, term, pos):
        if self.last_term_pos is None:
            self.last_term_pos = (term, pos)
            return False
        # If there is an adjacent adjective and noun in the query
        if wn.is_adverb(self.last_term_pos[1]) and wn.is_verb(pos):
            # Get term locations inside this document
            posting1 = self.vector_collection.get_term_posting_for_doc(term, self.doc.id)
            posting2 = self.vector_collection.get_term_posting_for_doc(self.last_term_pos[0], self.doc.id)
            # Determine if both terms appear in the document
            if posting1 is None or posting2 is None:
                return False
            # Boost if ADV and VERB appear in same sentence and same order as query
            sentences = zip(posting1.sentence, posting2.sentence)
            for idx, s1_s2 in enumerate(sentences):
                s1 = s1_s2[0]
                s2 = s1_s2[1]
                if s1 == s2 and posting1.offsets[idx] > posting2.offsets[idx]:
                    return True
        return False
