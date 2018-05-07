# Holds distance functions to compute distances between queries and documents
# Eric LaBouve (elabouve@calpoly.edu)
#
# DATASET = cran: query_limit=225, doc_limit=None, stemming_on=True
# Unmodified Cosine:                                                MAP=0.28156330866837304
# Unmodified Okapi:                                                 MAP=0.28192613853461285
# sub_all prob=0.10                                                 MAP=
# is_remove_adj                                                     MAP=0.2277293845592815
# is_remove_adv                                                     MAP=0.2813225565398429
#
# DATASET = adi: query_limit=35, doc_limit=None, stemming_on=True
# Unmodified Cosine:                                                MAP=0.37783666300710345
# Unmodified Okapi:                                                 MAP=0.3789677267181438
# sub_all prob=0.10                                                 MAP=
# is_remove_adj                                                     MAP=0.34522225013182856
# is_remove_adv                                                     MAP=0.38526070879256363
#
# DATASET = med: query_limit=30, doc_limit=None, stemming_on=True
# Unmodified Cosine:                                                MAP=0.5299083680454879
# Unmodified Okapi:                                                 MAP=0.5345275109699796
# sub_all prob=0.40                                                 MAP=0.5348136735114636
# is_remove_adj                                                     MAP=0.4985415092488889
# is_remove_adv                                                     MAP=0.535015149771215
#
#
#
#
#
# WILL NEED TO RECOMPUTE ALL THESE BECAUSE NOT ALL STOP WORDS WERE BEING FILTERED :(
#
# DATASET = cran: query_limit=225, doc_limit=None, stemming_on=True
# Unmodified Cosine:                                                MAP=0.280012462116408
# Unmodified Okapi:                                                 MAP=0.2811472330446242
# is_early_noun_adj I=2.4                                           MAP=0.2952156184300021
# is_early_noun_adj I=2.4, is_adj_noun_linear_pairs b=1.5           MAP=0.2959890583731966

# DATASET = adi: query_limit=35, doc_limit=None, stemming_on=True
# Unmodified Cosine:                                                MAP=0.3731938811079862
# Unmodified Okapi:                                                 MAP=0.3782581975783929
# is_early_noun_adj I=2.4                                           MAP=0.40435711100843064
# is_early_noun_adj I=2.4, is_adj_noun_linear_pairs b=1.5           MAP=0.40435711100843064

# DATASET = med: query_limit=30, doc_limit=None, stemming_on=True
# Unmodified Cosine:                                                MAP=0.5302462663875682
# Unmodified Okapi:                                                 MAP=0.5353128722733899
# is_early_noun_adj I=2.4                                           MAP=0.541351091646442
# is_early_noun_adj I=2.4, is_adj_noun_linear_pairs b=1.5           MAP=0.541373479735338


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
# doc_limit - An upper limit for the number of document ids returned per query.
#             Negative value indicates that all documents should be used.
def find_closest_docs(documents: VectorCollection, dist_obj: DistanceFunction, doc_limit=-1) -> list:
    ranked_docs = []  # Holds (doc id, distance)
    for docid, docvector in documents.id_to_textvector.items():
        dist_obj.set_doc(docvector)
        dist = dist_obj.execute()
        ranked_docs.append((docid, dist))
    ranked_docs.sort(key=lambda x: x[1], reverse=True)
    ranked_docs = [x[0] for x in ranked_docs]
    if doc_limit > 0:
        return ranked_docs[0:doc_limit]
    return ranked_docs


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


# Scales up when term is less frequent
def compute_idf(vector_collection: VectorCollection, term: str) -> float:
    dfi = vector_collection.get_doc_freq(term)
    num_docs = len(vector_collection.id_to_textvector)
    return math.log((num_docs - dfi + 0.5) / (dfi + 0.5))

# Flawed because we want our boosts to be independent of one another
#
# Positively boosts the Okapi score
# cur_score - Score to be boosted
# influence - Boosts the current score by value of influence
# def boost(cur_score: float, influence: float) -> float:
#     if cur_score > 0:
#         cur_score *= influence
#     else:
#         cur_score += influence * abs(cur_score) - abs(cur_score)
#     return cur_score


# Returns how much the current score should be boosted
def boost(cur_score: float, influence: float) -> float:
    if cur_score > 0:
        return (cur_score * influence) - cur_score
    else:
        return (abs(cur_score) * influence) - abs(cur_score)


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
        okapi_sum = 0
        for term, weight in self.query.term_to_freq.items():
            dfi = self.vector_collection.get_doc_freq(term)
            fij = self.doc.term_to_freq[term]
            dlj = len(self.doc)
            fiq = self.query.term_to_freq[term]

            first_term = math.log((self.num_docs - dfi + 0.5) / (dfi + 0.5))
            second_term = ((self.k1 + 1) * fij) / (self.k1 * (1 - self.b + self.b * dlj / self.avdl) + fij)
            third_term = ((self.k2 + 1) * fiq) / (self.k2 + fiq)

            product = first_term * second_term * third_term
            okapi_sum += product
        return okapi_sum


class OkapiModFunction(DistanceFunction):
    def __init__(self, vector_collection: VectorCollection,
                 is_early=False, is_early_noun=False, is_early_verb=False, is_early_adj=False, is_early_adv=False, early_term_influence=2.4,
                 is_early_noun_adj=False, is_early_verb_adv=False,
                 is_early_not_noun=False, is_early_not_verb=False, is_early_not_adj=False, is_early_not_adv=False,
                 is_early_not_verb_adv=False, is_early_not_noun_adj=False,

                 is_early_q=False, is_early_q_noun=False, is_early_q_verb=False,

                 is_close_pairs=False, close_pairs_influence=2.0,

                 is_noun=False, noun_influence=1.0,
                 is_verb=False, verb_influence=1.0,

                 is_adj_noun_pairs=False, adj_noun_pairs_influence=1.8, is_adj_noun_2gram=False, adj_noun_2gram_influence=2.0,
                 is_adj_noun_linear_pairs=False, adj_noun_pairs_m=-0.25, adj_noun_pairs_b=1.5,
                 is_adv_verb_pairs=False, adv_verb_pairs_influence=1.2,
                 is_adv_verb_linear_pairs=False, adv_verb_pairs_m=-0.25, adv_verb_pairs_b=1.25,

                 is_sub_all=False, is_sub_noun=False, is_sub_verb=False, is_sub_adj=False, is_sub_adv=False, sub_prob=0.25,
                 is_sub_idf=False, sub_idf_top=5,

                 is_remove_adj=False, is_remove_adv=False):

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
        self.is_early_noun = is_early_noun
        self.is_early_verb = is_early_verb
        self.is_early_adj = is_early_adj
        self.is_early_adv = is_early_adv
        self.early_term_influence = early_term_influence
        self.is_early_noun_adj = is_early_noun_adj
        self.is_early_verb_adv = is_early_verb_adv
        self.is_early_not_noun = is_early_not_noun
        self.is_early_not_verb = is_early_not_verb
        self.is_early_not_adj = is_early_not_adj
        self.is_early_not_adv = is_early_not_adv
        self.is_early_not_verb_adv = is_early_not_verb_adv
        self.is_early_not_noun_adj = is_early_not_noun_adj
        self.is_early_q = is_early_q
        self.is_early_q_noun = is_early_q_noun
        self.is_early_q_verb = is_early_q_verb
        # is_close_pairs variables
        self.is_close_pairs = is_close_pairs
        self.close_pairs_influence = close_pairs_influence
        self.last_term = None
        # is_noun variables
        self.is_noun = is_noun
        self.noun_influence = noun_influence
        # is_verb variables
        self.is_verb = is_verb
        self.verb_influence = verb_influence
        # is_adj_noun_pairs variables
        self.is_adj_noun_pairs = is_adj_noun_pairs
        self.is_adj_noun_2gram = is_adj_noun_2gram
        self.is_adj_noun_linear_pairs = is_adj_noun_linear_pairs
        self.adj_noun_pairs_influence = adj_noun_pairs_influence
        self.adj_noun_2gram_influence = adj_noun_2gram_influence
        self.adj_noun_pairs_m = adj_noun_pairs_m
        self.adj_noun_pairs_b = adj_noun_pairs_b
        # is_adj_noun_pairs variables
        self.is_adv_verb_pairs = is_adv_verb_pairs
        self.is_adv_verb_linear_pairs = is_adv_verb_linear_pairs
        self.adv_verb_pairs_influence = adv_verb_pairs_influence
        self.adv_verb_pairs_m = adv_verb_pairs_m
        self.adv_verb_pairs_b = adv_verb_pairs_b
        # Word substitutions from WordNet
        self.is_sub_all = is_sub_all
        self.is_sub_noun = is_sub_noun
        self.is_sub_verb = is_sub_verb
        self.is_sub_adj = is_sub_adj
        self.is_sub_adv = is_sub_adv
        self.sub_prob = sub_prob
        self.is_sub_idf = is_sub_idf
        self.sub_idf_top = sub_idf_top
        # remove variables
        self.is_remove_adj = is_remove_adj
            # Only include adj if next doc term matches next query term
        self.remove_adj_boosts = 0.0
        self.remove_last_term_adj = None
        self.is_remove_adv = is_remove_adv
            # Only include adv if next doc term matches next query term
        self.remove_adv_boosts = 0.0
        self.remove_last_term_adv = None

    def execute(self) -> float:
        okapi_sum = 0
        terms = []

        # Need to know idf scores ahead of time
        if self.is_sub_idf:
            is_top_idf_map = self.calc_top_idfs()

        # Traverse query terms in order of how they appear
        # Do not want to double score the same term
        for term, pos, subs in zip(self.query.terms, self.query.terms_pos, self.query.terms_sub):
            product = 0
            if term not in terms:
                product = self.okapi(term)

            boosts = []  # Independent collection of boosts
            sub_boosts = []  # Substitution boosts

            if self.is_early:
                boosts.append(boost(product, self.early_term(term)))
            if self.is_early_noun:
                if wn.is_noun(pos):
                    boosts.append(boost(product, self.early_term(term)))
            if self.is_early_verb:
                if wn.is_verb(pos):
                    boosts.append(boost(product, self.early_term(term)))
            if self.is_early_adj:
                if wn.is_adjective(pos):
                    boosts.append(boost(product, self.early_term(term)))
            if self.is_early_adv:
                if wn.is_adverb(pos):
                    boosts.append(boost(product, self.early_term(term)))
            if self.is_early_noun_adj: # Boost both adjectives and nouns
                if wn.is_noun(pos) or wn.is_adjective(pos):
                    boosts.append(boost(product, self.early_term(term)))
            if self.is_early_verb_adv: # Boost both verbs and adv
                if wn.is_verb(pos) or wn.is_adverb(pos):
                    boosts.append(boost(product, self.early_term(term)))
            if self.is_early_not_noun:
                if not wn.is_noun(pos):
                    boosts.append(boost(product, self.early_term(term)))
            if self.is_early_not_verb:
                if not wn.is_verb(pos):
                    boosts.append(boost(product, self.early_term(term)))
            if self.is_early_not_adj:
                if not wn.is_adjective(pos):
                    boosts.append(boost(product, self.early_term(term)))
            if self.is_early_not_adv:
                if not wn.is_adverb(pos):
                    boosts.append(boost(product, self.early_term(term)))
            if self.is_early_not_verb_adv:
                if not wn.is_verb(pos) and not wn.is_adverb(pos):
                    boosts.append(boost(product, self.early_term(term)))
            if self.is_early_not_noun_adj:
                if not wn.is_noun(pos) and not wn.is_adjective(pos):
                    boosts.append(boost(product, self.early_term(term)))

            if self.is_early_q:
                boosts.append(boost(product, self.early_term_q(term)))
            if self.is_early_q_noun:
                if wn.is_noun(pos):
                    boosts.append(boost(product, self.early_term_q(term)))
            if self.is_early_q_verb:
                if wn.is_verb(pos):
                    boosts.append(boost(product, self.early_term_q(term)))

            if self.is_noun:
                if wn.is_noun(pos):
                    boosts.append(boost(product, self.noun_influence))
            if self.is_verb:
                if wn.is_verb(pos):
                    boosts.append(boost(product, self.verb_influence))

            if self.is_close_pairs:
                boosts.append(boost(product, self.close_pairs(term)))
                self.last_term = term

            if self.is_adj_noun_pairs:
                # If the last adjective in the query is before the current noun
                if self.adj_noun_pairs(term, pos):
                    boosts.append(boost(product, self.adj_noun_pairs_influence))
                self.last_term_pos = (term, pos)
            if self.is_adj_noun_2gram:
                # If the last adjective in the query is right before the current noun
                if self.adj_noun_2gram(term, pos):
                    boosts.append(boost(product, self.adj_noun_2gram_influence))
                self.last_term_pos = (term, pos)
            if self.is_adj_noun_linear_pairs:
                boosts.append(boost(product, self.adj_noun_pairs_linear(term, pos)))
                self.last_term_pos = (term, pos)
            if self.is_adv_verb_pairs:
                # If the last adverb in the query is before the current verb
                if self.adv_verb_pairs(term, pos):
                    boosts.append(boost(product, self.adv_verb_pairs_influence))
                self.last_term_pos = (term, pos)
            if self.is_adv_verb_linear_pairs:
                boosts.append(boost(product, self.adv_verb_pairs_linear(term, pos)))
                self.last_term_pos = (term, pos)

            if self.is_sub_all:
                self.substitute(sub_boosts, subs)
            if self.is_sub_noun:
                if wn.is_noun(pos):
                    self.substitute(sub_boosts, subs)
            if self.is_sub_verb:
                if wn.is_verb(pos):
                    self.substitute(sub_boosts, subs)
            if self.is_sub_adj:
                if wn.is_adjective(pos):
                    self.substitute(sub_boosts, subs)
            if self.is_sub_adv:
                if wn.is_adverb(pos):
                    self.substitute(sub_boosts, subs)
            if self.is_sub_idf:  # substitute for the terms with the top idf scores
                if is_top_idf_map[term]:
                    self.substitute(sub_boosts, subs)

            if self.is_remove_adj:  # Needs to be last
                if wn.is_adjective(pos):
                    self.remove_adj_boosts = product + sum(boosts) + sum(sub_boosts)
                    self.remove_last_term_adj = term
                    terms.append(term)
                    continue
                # If we just found an adj and the next term is a noun found in both q and d
                elif wn.is_noun(pos) and self.remove_last_term_adj is not None\
                        and self.same_sentence(self.remove_last_term_adj, term):
                    boosts.append(self.remove_adj_boosts)
                    self.remove_adj_boosts = 0
                    self.remove_last_term_adj = None
                else:
                    self.remove_adj_boosts = 0
                    self.remove_last_term_adj = None
            if self.is_remove_adv:  # Needs to be last
                if wn.is_adverb(pos):
                    self.remove_adv_boosts = product + sum(boosts) + sum(sub_boosts)
                    self.remove_last_term_adv = term
                    terms.append(term)
                    continue
                # If we just found an adj and the next term is a noun found in both q and d
                elif wn.is_verb(pos) and self.remove_last_term_adv is not None\
                        and self.same_sentence(self.remove_last_term_adv, term):
                    boosts.append(self.remove_adv_boosts)
                    self.remove_adv_boosts = 0
                    self.remove_last_term_adv = None
                else:
                    self.remove_adv_boosts = 0
                    self.remove_last_term_adv = None

            terms.append(term)
            okapi_sum += product + sum(boosts) + sum(sub_boosts)
        return okapi_sum

    def substitute(self, sub_boosts, subs):
        for sub_term, prob in subs:
            if prob > self.sub_prob:
                weight = self.okapi(sub_term) * prob
                sub_boosts.append(weight)

    def okapi(self, term):
        dfi = self.vector_collection.get_doc_freq(term)
        fij = self.doc.term_to_freq[term]
        dlj = len(self.doc)
        fiq = self.query.term_to_freq[term]
        first_term = math.log((self.num_docs - dfi + 0.5) / (dfi + 0.5))
        second_term = ((self.k1 + 1) * fij) / (self.k1 * (1 - self.b + self.b * dlj / self.avdl) + fij)
        third_term = ((self.k2 + 1) * fiq) / (self.k2 + fiq)
        product = first_term * second_term * third_term
        return product

    def calc_top_idfs(self):
        term_idf = []
        for term in self.query.terms:
            dfi = self.vector_collection.get_doc_freq(term)
            idf = math.log((self.num_docs - dfi + 0.5) / (dfi + 0.5))
            term_idf.append((term, idf))
        top_idfs = sorted(term_idf, key=lambda x: x[1], reverse=True)
        if len(top_idfs) > self.sub_idf_top:
            top_idfs = top_idfs[0:self.sub_idf_top]
        is_top_idf = {}
        for term_idf_tup in term_idf:
            term = term_idf_tup[0]
            if term_idf_tup in top_idfs:
                is_top_idf[term] = True
            else:
                is_top_idf[term] = False
        return is_top_idf  # {"term":True/False} AND len(is_top_idf) == len(self.query.terms)


    # Boosts the query term's score if it is found earlier in the document
    # Compute as a percentage of the way through the document.
    # Range of values: [const, 1] where default const=2
    # boost = (2 * dl - term_loc) / dl
    # Best Case Scenario: boost(Term X X X) = 2
    # Worst Case Scenario: boost(X X X Term) ≈ 1 (Approaches 1 as sl gets large)
    # Example Best = (2 * 4 - 0) / 4 = 8 / 4 = 2
    # Example Worst = (2 * 4 - 3) / 4 = 5 / 4 = 1.25
    def early_term(self, term):
        dl = 1 if len(self.doc) == 0 else len(self.doc)
        posting = self.vector_collection.get_term_posting_for_doc(term, self.doc.id)
        term_loc = dl if posting is None else posting.offsets[0]
        return (self.early_term_influence * dl - term_loc) / dl

    # Boosts the query term's score if it is found earlier in the query
    # Compute as a percentage of the way through the query.
    # Range of values: [const, 1] where default const=2
    # boost = (2 * ql - term_loc) / ql
    # Best Case Scenario: boost(Term X X X) = 2
    # Worst Case Scenario: boost(X X X Term) ≈ 1 (Approaches 1 as sl gets large)
    # Example Best = (2 * 4 - 0) / 4 = 8 / 4 = 2
    # Example Worst = (2 * 4 - 3) / 4 = 5 / 4 = 1.25
    def early_term_q(self, term, const=2):
        ql = 1 if len(self.query) == 0 else len(self.query)
        term_loc = self.query.terms.index(term)
        return (const * ql - term_loc) / ql

    # Does not give good results. Gives too much weight to adjacent common words
    #   ex: 'high' 'speed', which detracts from the main subject of the query
    # MAP=0.23606341931968342
    # Gives an extra boost to adjacent query terms that are near each other
    # in the document. Idea is to give nonlinear reward for terms that are close
    # Evaluate only the closest pair of terms between the two postings.
    # Range of values: [2, 1]
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
        result = self.close_pairs_influence / min_dif
        return result

    # Gives an extra boost to adjectives and nouns that are found near each other
    # in the same sentence
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

    # Gives an extra boost to adjectives and nouns that are found right next to each other,
    def adj_noun_2gram(self, term, pos):
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
            # Boost if ADJ and NOUN appear in same sentence and right after each other as they do in the query
            sentences = zip(posting1.sentence, posting2.sentence)
            for idx, s1_s2 in enumerate(sentences):
                s1 = s1_s2[0]
                s2 = s1_s2[1]
                if s1 == s2 and ((posting1.offsets[idx] - posting2.offsets[idx]) == 1):
                    return True
        return False

    # Gives a linear boost to adjectives and nouns that are found near each other
    # in the same sentence. See __init__ for real m and b values.
    # y = mx + b
    # boost = max(m * (Idx(Nn)-Idx(Adj)) + (b-m), 1) and 0 ≤ m < -1
    #             m=-0.25                     m=-0.25
    #                                       b=2
    # Best Case Scenario: boost(X Adj Nn X) = 2
    # Middle Case Scenario: boost(adj X Nn X) =
    # Middle Case Scenario 2: boost(Adj X X Nn) ≈ 1 (Approaches 1 as separation gets larger)
    # Example Best = max(-0.25*(2-1) + (2 - -0.25), 1) = max(-0.25+2.25, 1) = 2
    # Example Middle = max(-0.25*(2-0) + (2 - -0.25), 1) = max(-.5+2.25, 1) = 1.75
    # Example Middle 2 = max(-0.25*(3-0) + (2 - -0.25), 1) = max(-0.75+2.25, 1) = 1.5
    # boost(Adj X X X X X X X Nn) = max(-0.25*(8-0) + (2 - -0.25), 1) = max(-2+2.25, 1) = 1
    def adj_noun_pairs_linear(self, term, pos):
        if self.last_term_pos is None:
            self.last_term_pos = (term, pos)
            return 1
        # If there is an adjacent adjective and noun in the query
        if wn.is_adjective(self.last_term_pos[1]) and wn.is_noun(pos):
            # Get term locations inside this document
            posting1 = self.vector_collection.get_term_posting_for_doc(term, self.doc.id)
            posting2 = self.vector_collection.get_term_posting_for_doc(self.last_term_pos[0], self.doc.id)
            # Determine if both terms appear in the document
            if posting1 is None or posting2 is None:
                return 1
            # Boost if ADJ and NOUN appear in same sentence and same order as query
            sentences = zip(posting1.sentence, posting2.sentence)
            for idx, s1_s2 in enumerate(sentences):
                s1 = s1_s2[0]
                s2 = s1_s2[1]
                if s1 == s2 and posting1.offsets[idx] > posting2.offsets[idx]:
                    m = self.adj_noun_pairs_m
                    b = self.adj_noun_pairs_b
                    idxNn = posting1.offsets[idx]
                    idxAdj = posting2.offsets[idx]
                    # print("Adj:" + str(self.last_term_pos[0]) + " Nn:" + term + " y:" + str(max(m * (idxNn-idxAdj) + (b-m), 1)) + " ___ ")
                    return max(m * (idxNn-idxAdj) + (b-m), 1)
        return 1

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

    # Gives an extra boost to adverbs and verbs that are found near each other
    # in the same sentence. See __init__ for m and b values.
    def adv_verb_pairs_linear(self, term, pos):
        if self.last_term_pos is None:
            self.last_term_pos = (term, pos)
            return 1
        # If there is an adjacent adjective and noun in the query
        if wn.is_adverb(self.last_term_pos[1]) and wn.is_verb(pos):
            # Get term locations inside this document
            posting1 = self.vector_collection.get_term_posting_for_doc(term, self.doc.id)
            posting2 = self.vector_collection.get_term_posting_for_doc(self.last_term_pos[0], self.doc.id)
            # Determine if both terms appear in the document
            if posting1 is None or posting2 is None:
                return 1
            # Boost if ADV and VERB appear in same sentence and same order as query
            sentences = zip(posting1.sentence, posting2.sentence)
            for idx, s1_s2 in enumerate(sentences):
                s1 = s1_s2[0]
                s2 = s1_s2[1]
                if s1 == s2 and posting1.offsets[idx] > posting2.offsets[idx]:
                    m = self.adv_verb_pairs_m
                    b = self.adv_verb_pairs_b
                    idxVb = posting1.offsets[idx]
                    idxAdv = posting2.offsets[idx]
                    return max(m * (idxVb - idxAdv) + (b - m), 1)
        return 1

    # Determines if two terms are found in order in the same sentence
    def same_sentence(self, term1, term2) -> bool:
        # Get term locations inside this document
        posting1 = self.vector_collection.get_term_posting_for_doc(term1, self.doc.id)
        posting2 = self.vector_collection.get_term_posting_for_doc(term2, self.doc.id)
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

