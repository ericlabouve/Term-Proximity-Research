from nltk.corpus import wordnet as wn
import json
import nltk


class WordNet:

    pos_str_map = {'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
                  'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n',
                  'RB': 'r', 'RBR': 'r', 'RBS': 'r',
                  'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v',
                  'VBP': 'v', 'VBZ': 'v'}

    def __init__(self):
        pass

    # Get synonyms that are the same part of speech to the original term
    # term - The word to get synonyms of and the part of speech
    # threshold - Only add synonyms if similarity between original term
    # and the synonym is above the threshold
    def get_syns(self, term_pos: (str, str), threshold=.5) -> list:
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

"""
        self.json_base_dir = '/Users/Eric/Desktop/Thesis/programs/java/json/'
        self.adj_list_json = json.load(open(self.json_base_dir + 'adjList.json'))
        self.edge_list_json = json.load(open(self.json_base_dir + 'edgeList.json'))
        self.id_to_label_json = json.load(open(self.json_base_dir + 'idToLabel.json'))
        self.label_to_id_json = json.load(open(self.json_base_dir + 'labelToId.json'))
        self.wf_vertex_db_json = json.load(open(self.json_base_dir + 'wfVertexDb.json'))

    def get_sim_terms(self, term: str) -> list:
        sim_terms = []
        if term in self.label_to_id_json:
            term_id = str(self.label_to_id_json[term])
            if term_id in self.adj_list_json:
                adj_nodes = self.adj_list_json[term_id]
                for adj_node_id in [str(x) for x in adj_nodes]:
                    if adj_node_id in self.id_to_label_json:
                        phrase = self.id_to_label_json[adj_node_id]
                        edge_list_key = str(term_id) + '-' + str(adj_node_id)
                        if edge_list_key in self.edge_list_json:
                            edge_weight = self.edge_list_json[edge_list_key]
                            sim_terms.append((phrase, edge_weight))
        return sim_terms
"""
