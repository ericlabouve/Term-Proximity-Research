from nltk.corpus import wordnet as wn
from collections import defaultdict
import json, random


def is_noun(tag: str) -> bool:
    return any(symbol in tag for symbol in ['NN', 'NNS', 'NNP', 'NNPS'])


def is_adjective(tag: str) -> bool:
    return any(symbol in tag for symbol in ['JJ', 'JJR', 'JJS'])


def is_adverb(tag: str) -> bool:
    return any(symbol in tag for symbol in ['RB', 'RBR', 'RBS'])


def is_verb(tag: str) -> bool:
    return any(symbol in tag for symbol in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])


class WordNet:

    pos_str_map = {'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
                  'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n',
                  'RB': 'r', 'RBR': 'r', 'RBS': 'r',
                  'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v',
                  'VBP': 'v', 'VBZ': 'v'}

    def __init__(self):
        self.json_base_dir = '../json/'
        self.adj_list_json = json.load(open(self.json_base_dir + 'adjList.json'))
        self.edge_list_json = json.load(open(self.json_base_dir + 'edgeList.json'))
        self.id_to_label_json = json.load(open(self.json_base_dir + 'idToLabel.json'))
        self.label_to_id_json = json.load(open(self.json_base_dir + 'labelToId.json'))
        self.wf_vertex_db_json = json.load(open(self.json_base_dir + 'wfVertexDb.json'))

    # Computes the similarity between two terms as the probability of reaching term2 from term1 and reaching
    # term1 from term1. The equation is: [P(term1 | term2) + P(term1 | term2)] / 2
    # P(a | b) is how many times in our iterations do we come across term a when starting at term b
    def compute_sim_rw(self, term1: str, term2: str, depth=5, iterations=1000) -> float:
        assert depth > 0 and iterations > 0

        def _random_walk(_term1: str, _term2: str) -> float:
            freq = 0  # How many times we discover term1
            # Assign a probability of zero if one of the terms is not in WordNet
            if _term1 not in self.label_to_id_json or _term2 not in self.label_to_id_json:
                return 0
            # Assign a probability of one if the two terms are the same
            if _term1 == _term2:
                return 1
            term1_id = self.label_to_id_json[_term1]
            # Run simulation iterations number of times
            for i in range(iterations):
                marked_node_ids = []
                sense_id = self.label_to_id_json[_term2]  # Current node in WordNet
                for j in range(depth):
                    # Get adjacent sense ids
                    all_adj_sense_ids = self.adj_list_json[str(sense_id)]
                    # Remove ids that have already been marked
                    adj_sense_ids = [x for x in all_adj_sense_ids if x not in marked_node_ids]
                    # Mark sense ids as seen
                    marked_node_ids += adj_sense_ids
                    # Get edge ids for unseen adj senses ------- CORRECT ORDER? -------
                    edge_ids = [str(sense_id) + '-' + str(x) for x in adj_sense_ids]  # From sense to adj_sense
                    # Get edge weights (similarity) between sense and adj_sense
                    edge_weights = [float(self.edge_list_json[x]) for x in edge_ids]
                    # Normalize all edge weights to be values between 0 and 1
                    sum = 0
                    for x in edge_weights:
                        sum += x
                    edge_weights_norm = [x/sum for x in edge_weights]
                    # Compute a random value between 0 and 1
                    val = random.uniform(0, 1)
                    # Determine which edge to traverse. Edge weights are traversal probabilities
                    for node_id, prob in zip(adj_sense_ids, edge_weights_norm):
                        val -= prob
                        if val <= 0:
                            sense_id = node_id
                            break
                    if sense_id == term1_id:
                        freq += 1
            return freq / iterations
        return (_random_walk(term1, term2) + _random_walk(term2, term1)) / 2

    # Obtains senses that rank highly when a random walk is computed
    def get_sim_terms_rw(self, term: str, depth=2, str_len=1, iterations=1000) -> list:
        assert depth > 0 and str_len > 0 and iterations > 0
        freq = defaultdict(int)  # How many times we discover a sense
        # Assign a probability of zero if one of the terms is not in WordNet
        if term not in self.label_to_id_json:
            return []
        term_id = self.label_to_id_json[term]
        # Run simulation iterations number of times
        for i in range(iterations):
            marked_node_ids = [term_id]
            sense_id = term_id  # Current node in WordNet
            for j in range(depth):
                # Get adjacent sense ids
                all_adj_sense_ids = self.adj_list_json[str(sense_id)]
                # Remove ids that have already been marked
                adj_sense_ids = [x for x in all_adj_sense_ids if x not in marked_node_ids]
                # Mark sense ids as seen
                marked_node_ids += adj_sense_ids
                # Get edge ids for unseen adj senses ------- CORRECT ORDER? -------
                edge_ids = [str(sense_id) + '-' + str(x) for x in adj_sense_ids]  # From sense to adj_sense
                # Get edge weights (similarity) between sense and adj_sense
                edge_weights = [float(self.edge_list_json[x]) for x in edge_ids]
                # Normalize all edge weights to be values between 0 and 1
                sum = 0
                for x in edge_weights:
                    sum += x
                edge_weights_norm = [x / sum for x in edge_weights]
                # Compute a random value between 0 and 1
                val = random.uniform(0, 1)
                # Determine which edge to traverse. Edge weights are traversal probabilities
                for node_id, prob in zip(adj_sense_ids, edge_weights_norm):
                    val -= prob
                    if val <= 0:
                        sense_id = node_id
                        break
                # Record frequency of node we land on
                freq[sense_id] += 1

        norm_term = 0
        # Get normalization term for next loop
        for sense_id, frq in freq.items():
            try:
                sense = self.id_to_label_json[str(sense_id)]
                if len(sense.split()) <= str_len:  # one word
                    norm_term += frq
            except KeyError:
                pass

        # Normalize all terms
        sim_terms = []
        for sense_id, frq in freq.items():
            sense = self.id_to_label_json[str(sense_id)]
            if len(sense.split()) <= str_len:  # one word
                sim_terms.append((sense, frq / norm_term))

        sim_terms.sort(key=lambda x: x[1], reverse=True)
        return sim_terms

    # Obtains all senses for the given term in the WordNet graph
    # depth is how many edges to traverse away from term and must be at least 1
    # Returns an array of senses with their rough similarities to the original term
    def get_sim_terms(self, term: str, depth=2, str_len=1):
        assert depth > 0 and str_len > 0
        marked_node_ids = []
        sim_terms = []
        queue_this = []  # Terms for the current iteration
        queue_next = []  # Terms for the next iteration
        # Setup
        queue_next += [(term, 1.0)]
        if term in self.label_to_id_json:
            marked_node_ids += [self.label_to_id_json[term]]
        # Loop till depth number of iterations
        for i in range(depth):
            queue_this += queue_next
            for sense, similarity in queue_this:
                try:
                    # Get sense id
                    sense_id = str(self.label_to_id_json[sense])
                    # Get adjacent sense ids
                    all_adj_sense_ids = self.adj_list_json[sense_id]
                    # Get adjacent senses and ids that have not already been marked
                    adj_sense_ids = [x for x in all_adj_sense_ids if x not in marked_node_ids]
                    adj_senses = [self.id_to_label_json[str(x)] for x in adj_sense_ids]
                    # Mark sense ids as seen
                    marked_node_ids += adj_sense_ids
                    # Get edge ids for unseen adj senses ------- CORRECT ORDER? -------
                    edge_ids = [str(sense_id)+'-'+str(x) for x in adj_sense_ids] # From sense to adj_sense
                    # Get edge weights (similarity) between sense and adj_sense
                    edge_weights = [self.edge_list_json[x] for x in edge_ids]
                    # Normalize all edge weights to be values between 0 and 1
                    sum = 0
                    for x in edge_weights:
                        sum += float(x)
                    edge_weights_norm = [float(x) / sum for x in edge_weights]
                    # Compute rough similarity between original term and this sense
                    sense_sim_tup = [(sense, float(sim) * float(similarity)) for sense, sim in zip(adj_senses, edge_weights_norm)]
                    # Add senses to the queue to be computed later
                    queue_next += sense_sim_tup
                    sim_terms += [tup for tup in sense_sim_tup if len(tup[0].split()) <= str_len]
                except KeyError as e:
                    pass
        sim_terms.sort(key=lambda x: x[1], reverse=True)
        return sim_terms

    # Get synonyms that are the same part of speech to the original term. Does not compute similarity score
    # term - The word to get synonyms of and the part of speech
    # threshold - Only add synonyms if similarity between original term
    # and the synonym is above the threshold
    def get_syns(self, term_pos: (str, str)) -> list:
        synonyms = []
        term = term_pos[0]
        # Include only words with same part of speech
        if term_pos[1] in self.pos_str_map:
            pos = self.pos_str_map[term_pos[1]]
            synsets = wn.synsets(term, pos=pos)
        else:
            synsets = wn.synsets(term_pos[0])
        for synset in synsets:
            for lemma in synset.lemmas():
                word = lemma.name()
                # Words are not similar unless they are 100% similar
                # This value should always be 1 if they are the same p.o.s.
                if synset.wup_similarity(lemma._synset) == 1:
                    # Dont include phrases
                    ngram = True if '_' in word or '-' in word else False
                    if word not in synonyms and not ngram:
                        synonyms.append(word)
        return synonyms

    # Stems all terms in the term_prob tuple list
    # Combines tuples that stem to the same term
    # Makes sure that terms are not stemmed to root_term
    def stem(self, stemmer, root_term, term_prob: list):
        if root_term == 'humans':
            pass
        d = defaultdict(float)
        l = []
        ln = []
        root_term = stemmer.stem(root_term)
        # Stem all terms
        for term, prob in term_prob:
            stem = stemmer.stem(term)
            d[stem] += prob

        # Create (term, probability) list
        for key, value in d.items():
            if key != root_term:
                l += [(key, value)]

        probs = [x[1] for x in l]
        nsum = sum(probs)
        # Normalize probabilities to add to one because some terms stem to the same value or the root term
        for term, prob in l:
            ln += [(term, prob / nsum)]
        ln.sort(key=lambda x: x[1], reverse=True)
        return ln


"""
POS tag list:

CC	    coordinating conjunction
CD	    cardinal digit
DT	    determiner
EX	    existential there (like: "there is" ... think of it like "there exists")
FW	    foreign word
IN	    preposition/subordinating conjunction
JJ	    adjective	'big'
JJR	    adjective, comparative	'bigger'
JJS	    adjective, superlative	'biggest'
LS	    list marker	1)
MD	    modal	could, will
NN	    noun, singular 'desk'
NNS	    noun plural	'desks'
NNP	    proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	    predeterminer	'all the kids'
POS	    possessive ending	parent's
PRP	    personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	    adverb	very, silently,
RBR	    adverb, comparative	better
RBS	    adverb, superlative	best
RP	    particle	give up
TO	    to	go 'to' the store.
UH	    interjection	errrrrrrrm
VB	    verb, base form	take
VBD	    verb, past tense	took
VBG	    verb, gerund/present participle	taking
VBN	    verb, past participle	taken
VBP	    verb, sing. present, non-3d	take
VBZ	    verb, 3rd person sing. present	takes
WDT	    wh-determiner	which
WP	    wh-pronoun	who, what
WP$	    possessive wh-pronoun	whose
WRB	    wh-abverb	where, when
"""
