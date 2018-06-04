import json

class Word2Vec:
    def __init__(self):
        self.json_base_dir = './'
        self.sim_json = json.load(open(self.json_base_dir + 'substitutions_word2vec.json'))
        self.mean = 0.5746  # Just for reference
        self.stdev = 0.0950  # Just for reference

    def get(self, term: str, minc=None) -> list:
        if term not in self.sim_json:
            return []
        sim = self.sim_json[term]
        if min is None:
            return sim
        else:
            return [x for x in sim if float(x[1]) >= minc]