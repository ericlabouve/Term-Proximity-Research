# Experiments and the location of main
# Eric LaBouve (elabouve@calpoly.edu)

import ScoringFunctions as score_fs
import nltk, json, sys
from VectorCollection import VectorCollection, VectorType
from DistanceFunctions import CosineFunction, OkapiFunction, OkapiModFunction, compute_idf
from WordNet import WordNet, is_noun, is_verb, is_adjective, is_adverb
from nltk.stem import PorterStemmer
from multiprocessing import Process, Queue, Array, Manager


# Run the func, save the results to the file indicated by the path, and save the MAP score
def run_save(func, func_name):
    results = qrys.find_closest_docs(docs, func, doc_limit=doc_limit, query_limit=query_limit)
    avg_map = score_fs.compute_avg_map(results, relevant_docs)
    results_file = out_dir + func_name + "_results.json"
    with open(results_file, 'w') as f:
        f.write(json.dumps(results))
    map_file = out_dir + func_name + "_map.txt"
    with open(map_file, 'w') as f:
        f.write(str(avg_map))
    print("\n" + func_name + ": " + str(avg_map))


def save(func_name, results, avg_map):
    results_file = out_dir + func_name + "_results.json"
    with open(results_file, 'w') as f:
        f.write(json.dumps(results))
    map_file = out_dir + func_name + "_map.txt"
    with open(map_file, 'w') as f:
        f.write(str(avg_map))
    print("\n-->" + func_name + ": " + str(avg_map))




# _____________Functions for reading documents, queries and relevant documents_____________

def read_cran(path, title=True):
    if title:
        docs = VectorCollection(path + "/cran/cran.all.1400", VectorType.DOCUMENTS, stemming_on=True)
    else:
        docs = VectorCollection(path + "/cran/cran.notitle.all.1400", VectorType.DOCUMENTS, stemming_on=True)
    qrys = VectorCollection(path + "/cran/cran.qry", VectorType.QUERIES, stemming_on=True)
    relevant_docs = score_fs.read_human_judgement(path + "/cran/cranqrel", 1, 3)
    return docs, qrys, relevant_docs, "cran/"
def read_adi(path):
    docs = VectorCollection(path + "/adi/ADI.ALL", VectorType.DOCUMENTS, stemming_on=True)
    qrys = VectorCollection(path + "/adi/ADI.QRY", VectorType.QUERIES, stemming_on=True)
    relevant_docs = score_fs.read_human_judgement(path + "/adi/ADI.REL", 0, 0)  # DIFFERENT FORMAT
    return docs, qrys, relevant_docs, "adi/"
def read_med(path):
    docs = VectorCollection(path + "/med/MED.ALL", VectorType.DOCUMENTS, stemming_on=True)
    qrys = VectorCollection(path + "/med/MED.QRY", VectorType.QUERIES, stemming_on=True)
    relevant_docs = score_fs.read_human_judgement_MED(path + "/med/MED.REL", 1, 1)  # DIFFERENT FORMAT
    return docs, qrys, relevant_docs, "med/"
def read_time(path):
    docs = VectorCollection(path + "/time/TIME_clean.ALL", VectorType.DOCUMENTS, stemming_on=True)
    qrys = VectorCollection(path + "/time/TIME_clean.QUE", VectorType.QUERIES, stemming_on=True)
    relevant_docs = score_fs.read_human_judgement_TIME(path + "/time/TIME_clean.REL")
    return docs, qrys, relevant_docs, "time/"

# __________________________ Tests _______________________________
def test_cosine():
    docs.normalize(docs)
    qrys.normalize(docs)
    cos_func = CosineFunction(docs)
    run_save(cos_func, "cosine")
def test_okapi():
    okapi_func = OkapiFunction(docs)
    run_save(okapi_func, "okapi")
def test_is_remove_adj():
    okapi_func = OkapiModFunction(docs, is_remove_adj=True)
    run_save(okapi_func, "remove adj")
def test_is_remove_adv():
    okapi_func = OkapiModFunction(docs, is_remove_adv=True)
    run_save(okapi_func, "remove adv")

def test_sub_all():
    okapi_func = OkapiModFunction(docs, is_sub_all=True, sub_prob=0.25)
    run_save(okapi_func, "sub all i=.25")
def test_sub_noun():
    okapi_func = OkapiModFunction(docs, is_sub_noun=True, sub_prob=0.25)
    run_save(okapi_func, "sub noun i=.25")
def test_sub_verb():
    okapi_func = OkapiModFunction(docs, is_sub_verb=True, sub_prob=0.25)
    run_save(okapi_func, "sub verb i=.25")
def test_sub_adj():
    okapi_func = OkapiModFunction(docs, is_sub_adj=True, sub_prob=0.25)
    run_save(okapi_func, "sub adj i=.25")
def test_sub_adv():
    okapi_func = OkapiModFunction(docs, is_sub_adv=True, sub_prob=0.25)
    run_save(okapi_func, "sub adv i=.25")
def test_sub_idf_top():
    okapi_func = OkapiModFunction(docs, is_sub_idf_top=True, sub_prob=0.25, sub_idf_top=5)
    run_save(okapi_func, "sub idf top=5 i=.25")
def test_sub_idf_bottom():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.25, sub_idf_bottom=5)
    run_save(okapi_func, "sub idf bottom=5 i=.25")

def test_early_all():
    okapi_func = OkapiModFunction(docs, is_early=True, early_term_influence=1.4)
    run_save(okapi_func, "early all i=1.4")
def test_early_noun():
    okapi_func = OkapiModFunction(docs, is_early_noun=True, early_term_influence=1.2)
    run_save(okapi_func, "early noun i=1.2")
def test_early_verb():
    okapi_func = OkapiModFunction(docs, is_early_verb=True, early_term_influence=1.2)
    run_save(okapi_func, "early verb i=1.2")
def test_early_adj():
    okapi_func = OkapiModFunction(docs, is_early_adj=True, early_term_influence=1.6)
    run_save(okapi_func, "early adj i=1.6")
def test_early_adv():
    okapi_func = OkapiModFunction(docs, is_early_adv=True, early_term_influence=1.2)
    run_save(okapi_func, "early adv i=1.2")
def test_early_noun_adj():
    okapi_func = OkapiModFunction(docs, is_early_noun_adj=True, early_term_influence=1.6)
    run_save(okapi_func, "early noun adj i=1.6")
def test_early_verb_adv():
    okapi_func = OkapiModFunction(docs, is_early_verb_adv=True, early_term_influence=1.4)
    run_save(okapi_func, "early verb adv i=1.4")
def test_early_not_noun():
    okapi_func = OkapiModFunction(docs, is_early_not_noun=True, early_term_influence=1.4)
    run_save(okapi_func, "early not noun i=1.4")
def test_early_not_verb():
    okapi_func = OkapiModFunction(docs, is_early_not_verb=True, early_term_influence=1.8)
    run_save(okapi_func, "early not verb i=1.8")
def test_early_not_adj():
    okapi_func = OkapiModFunction(docs, is_early_not_adj=True, early_term_influence=1.2)
    run_save(okapi_func, "early not adj i=1.2")
def test_early_not_adv():
    okapi_func = OkapiModFunction(docs, is_early_not_adv=True, early_term_influence=2.8)
    run_save(okapi_func, "early not adv i=2.8")
def test_early_not_verb_adv():
    okapi_func = OkapiModFunction(docs, is_early_not_verb_adv=True, early_term_influence=1.8)
    run_save(okapi_func, "early not verb adv i=1.8")
def test_early_not_noun_adj():
    okapi_func = OkapiModFunction(docs, is_early_not_noun_adj=True, early_term_influence=1.2)
    run_save(okapi_func, "early not noun adj i=1.2")

def test_noun():
    okapi_func = OkapiModFunction(docs, is_noun=True, noun_influence=1.4)
    run_save(okapi_func, "noun i=1.4")
def test_adj():
    okapi_func = OkapiModFunction(docs, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "adj i=1.2")
def test_verb():
    okapi_func = OkapiModFunction(docs, is_verb=True, verb_influence=1.2)
    run_save(okapi_func, "verb i=1.2")
def test_adv():
    okapi_func = OkapiModFunction(docs, is_adv=True, adv_influence=2.8)
    run_save(okapi_func, "adv i=2.8")
def test_noun2():
    okapi_func = OkapiModFunction(docs, is_noun=True, noun_influence=0.8)
    run_save(okapi_func, "noun i=0.8")
def test_adj2():
    okapi_func = OkapiModFunction(docs, is_adj=True, adj_influence=0.8)
    run_save(okapi_func, "adj i=0.8")
def test_verb2():
    okapi_func = OkapiModFunction(docs, is_verb=True, verb_influence=0.4)
    run_save(okapi_func, "verb i=0.4")
def test_adv2():
    okapi_func = OkapiModFunction(docs, is_adv=True, adv_influence=0.8)
    run_save(okapi_func, "adv i=0.8")

def test_is_close_all():
    okapi_func = OkapiModFunction(docs, is_close_pairs=True, close_pairs_m=-.25, close_pairs_b=1.25)
    run_save(okapi_func, "close_pairs i=1.25")
def test_is_adj_noun_linear_pairs():
    okapi_func = OkapiModFunction(docs, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=2.0)
    run_save(okapi_func, "adj_noun_linear_pairs i=2.0")
def test_is_adv_verb_linear_pairs():
    okapi_func = OkapiModFunction(docs, is_adv_verb_linear_pairs=True, adv_verb_pairs_m=-.25, adv_verb_pairs_b=1.25)
    run_save(okapi_func, "adv_verb_linear_pairs i=1.25")

def test_bigrams():
    okapi_func = OkapiModFunction(docs, is_bigram=True, bigram_influence=1.2)
    run_save(okapi_func, "bigrams i=1.2")
def test_adj_noun_bigrams():
    okapi_func = OkapiModFunction(docs, is_adj_noun_2gram=True, adj_noun_2gram_influence=2.0)
    run_save(okapi_func, "adj_noun_bigrams i=2.0")
def test_adv_verb_bigrams():
    okapi_func = OkapiModFunction(docs, is_adv_verb_2gram=True, adv_verb_2gram_influence=1.2)
    run_save(okapi_func, "adv_verb_bigrams i=1.2")

# __________________________ Train Linear _______________________________
def train_sub_all():
    influence = 0.02
    funcs = []
    while influence <= 0.1:
        func = OkapiModFunction(docs, is_sub_all=True, sub_prob=influence)
        funcs.append((func, "sub_all" + " i=" + str(influence)))
        influence += 0.02
    run_funcs(funcs)
def train_sub_noun():
    influence = 0.02
    funcs = []
    while influence <= 0.1:
        func = OkapiModFunction(docs, is_sub_noun=True, sub_prob=influence)
        funcs.append((func, "sub_noun" + " i=" + str(influence)))
        influence += 0.02
    run_funcs(funcs)
def train_sub_verb():
    influence = 0.02
    funcs = []
    while influence <= 0.1:
        func = OkapiModFunction(docs, is_sub_verb=True, sub_prob=influence)
        funcs.append((func, "sub_verb" + " i=" + str(influence)))
        influence += 0.02
    run_funcs(funcs)
def train_sub_adj():
    influence = 0.02
    funcs = []
    while influence <= 0.1:
        func = OkapiModFunction(docs, is_sub_adj=True, sub_prob=influence)
        funcs.append((func, "sub_adj" + " i=" + str(influence)))
        influence += 0.02
    run_funcs(funcs)
def train_sub_adv():
    influence = 0.02
    funcs = []
    while influence <= 0.1:
        func = OkapiModFunction(docs, is_sub_adv=True, sub_prob=influence)
        funcs.append((func, "sub_adv" + " i=" + str(influence)))
        influence += 0.02
    run_funcs(funcs)
def train_sub_idf_top():
    influence = 0.02
    funcs = []
    while influence <= 0.1:
        func = OkapiModFunction(docs, is_sub_idf_top=True, sub_prob=influence, sub_idf_top=5)
        funcs.append((func, "sub_idf top=5" + " i=" + str(influence)))
        influence += 0.02
    run_funcs(funcs)
def train_sub_idf_bottom():
    influence = 0.02
    funcs = []
    while influence <= 0.1:
        func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=influence, sub_idf_bottom=5)
        funcs.append((func, "sub_idf bottom=5" + " i=" + str(influence)))
        influence += 0.02
    run_funcs(funcs)

def train_is_early_all():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_early=True, early_term_influence=influence)
        funcs.append((func, "early_all" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_is_early_noun():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_early_noun=True, early_term_influence=influence)
        funcs.append((func, "early_noun" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_is_early_adj():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_early_adj=True, early_term_influence=influence)
        funcs.append((func, "early_adj" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_is_early_verb():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_early_verb=True, early_term_influence=influence)
        funcs.append((func, "early_verb" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_is_early_adv():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_early_adv=True, early_term_influence=influence)
        funcs.append((func, "early_adv" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_is_early_noun_adj():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_early_noun_adj=True, early_term_influence=influence)
        funcs.append((func, "early_noun_adj" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_is_early_verb_adv():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_early_verb_adv=True, early_term_influence=influence)
        funcs.append((func, "early_verb_adv" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)

def train_early_not_noun():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_early_not_noun=True, early_term_influence=influence)
        funcs.append((func, "early_not_noun" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_early_not_verb():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_early_not_verb=True, early_term_influence=influence)
        funcs.append((func, "early_not_verb" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_early_not_adj():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_early_not_adj=True, early_term_influence=influence)
        funcs.append((func, "early_not_adj" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_early_not_adv():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_early_not_adv=True, early_term_influence=influence)
        funcs.append((func, "early_not_adv" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_early_not_verb_adv():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_early_not_verb_adv=True, early_term_influence=influence)
        funcs.append((func, "early_not_verb_adv" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_early_not_noun_adj():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_early_not_noun_adj=True, early_term_influence=influence)
        funcs.append((func, "early_not_noun_adj" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)

def train_noun():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_noun=True, noun_influence=influence)
        funcs.append((func, "noun" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_adj():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_adj=True, adj_influence=influence)
        funcs.append((func, "adj" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_verb():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_verb=True, verb_influence=influence)
        funcs.append((func, "verb" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_adv():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_adv=True, adv_influence=influence)
        funcs.append((func, "adv" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_noun2():
    influence = .2
    funcs = []
    while influence <= 0.8:
        func = OkapiModFunction(docs, is_noun=True, noun_influence=influence)
        funcs.append((func, "noun" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_adj2():
    influence = .2
    funcs = []
    while influence <= 0.8:
        func = OkapiModFunction(docs, is_adj=True, adj_influence=influence)
        funcs.append((func, "adj" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_verb2():
    influence = .2
    funcs = []
    while influence <= 0.8:
        func = OkapiModFunction(docs, is_verb=True, verb_influence=influence)
        funcs.append((func, "verb" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_adv2():
    influence = .2
    funcs = []
    while influence <= 0.8:
        func = OkapiModFunction(docs, is_adv=True, adv_influence=influence)
        funcs.append((func, "adv" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)

def train_is_close_all():
    influence = 1.25
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_close_pairs=True, close_pairs_m=-.25, close_pairs_b=influence)
        funcs.append((func, "close_pairs" + " i=" + str(influence)))
        influence += 0.25
    run_funcs(funcs)
def train_is_adj_noun_linear_pairs():
    influence = 1.25
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=influence)
        funcs.append((func, "adj_noun_linear_pairs" + " i=" + str(influence)))
        influence += 0.25
    run_funcs(funcs)
def train_is_adv_verb_linear_pairs():
    influence = 1.25
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_adv_verb_linear_pairs=True, adv_verb_pairs_m=-.25, adv_verb_pairs_b=influence)
        funcs.append((func, "adv_verb_linear_pairs" + " i=" + str(influence)))
        influence += 0.25
    run_funcs(funcs)

def train_bigrams():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_bigram=True, bigram_influence=influence)
        funcs.append((func, "bigrams" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_is_adj_noun_bigrams():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_adj_noun_2gram=True, adj_noun_2gram_influence=influence)
        funcs.append((func, "adj_noun_bigrams" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)
def train_is_adv_verb_bigrams():
    influence = 1.2
    funcs = []
    while influence <= 3.0:
        func = OkapiModFunction(docs, is_adv_verb_2gram=True, adv_verb_2gram_influence=influence)
        funcs.append((func, "adv_verb_bigrams" + " i=" + str(influence)))
        influence += 0.2
    run_funcs(funcs)

def run_funcs(funcs):
    best_map = 0
    best_result = None
    best_label = None
    for func, label in funcs:
        results = qrys.find_closest_docs(docs, func, doc_limit=doc_limit, query_limit=query_limit)
        avg_map = score_fs.compute_avg_map(results, relevant_docs)
        print("\n" + label + ": " + str(avg_map))
        if (avg_map > best_map):
            best_map = avg_map
            best_result = results
            best_label = label
    save(best_label, best_result, best_map)


# __________________________ Train Multiprocess _______________________________
def run(queue, func, label):
    results = qrys.find_closest_docs(docs, func, doc_limit=doc_limit, query_limit=query_limit)
    avg_map = score_fs.compute_avg_map(results, relevant_docs)
    queue.append((label, avg_map, results))
def start_processes(process_list, qu):
    # Start all processes
    for p in process_list:
        p.start()

    # Wait for all processes to end
    for p in process_list:
        p.join()

    # Sort based on base MAP score and display stats
    q = sorted(qu, key=lambda x: x[1], reverse=True)
    print("\n" + str(q))
    label, avg_map, results = q[0]
    save(label, results, avg_map)

def train_sub_all_old():
    # Loop for running processes in parallel that differ by level of influence
    m = Manager()
    qu = m.list()
    process_list = []
    influence = 0.02
    while influence <= 0.1:
        func = OkapiModFunction(docs, is_sub_all=True, sub_prob=influence)
        p = Process(target=run, args=(qu, func, 'sub_all prob=' + str(influence)))
        process_list.append(p)
        influence += 0.02
    start_processes(process_list, qu)


# _________________________________ Main for Testing and Training ______________________________________
if __name__ == "__main__":
    abs_path = "/Users/Eric/Desktop/Thesis/projects/datasets"
    rel_path = "../datasets"
    f_path = abs_path
    docs, qrys, relevant_docs, dir = read_cran(f_path, title=False)
#    docs, qrys, relevant_docs, dir = read_adi(f_path)
#    docs, qrys, relevant_docs, dir = read_med(f_path)
#    docs, qrys, relevant_docs, dir = read_time(f_path)

    query_limit = -1  # Use all queries
    doc_limit = -1  # Use all documents
    out_dir = "out/train_cran/" + dir

    # ___________ Tests ___________
    # test_cosine()
    # test_okapi()
    #
    # test_is_remove_adj()
    # test_is_remove_adv()

    # test_sub_all()
    # test_sub_noun()
    # test_sub_verb()
    # test_sub_adj()
    # test_sub_adv()
    # test_sub_idf_top()
    # test_sub_idf_bottom()

    # test_early_all()
    # test_early_noun()
    # test_early_verb()
    # test_early_adj()
    # test_early_adv()
    # test_early_noun_adj()
    # test_early_verb_adv()
    # test_early_not_noun()
    # test_early_not_verb()
    # test_early_not_adj()
    # test_early_not_adv()
    # test_early_not_verb_adv()
    # test_early_not_noun_adj()

    # test_noun()
    # test_adj()
    # test_verb()
    # test_adv()
    # test_noun2()
    # test_adj2()
    # test_verb2()
    # test_adv2()

    # test_is_close_all()
    # test_is_adj_noun_linear_pairs()
    # test_is_adv_verb_linear_pairs()

    # test_bigrams()
    # test_adj_noun_bigrams()
    # test_adv_verb_bigrams()

    # ___________ Training ___________
    # train_sub_all()
    # train_sub_noun()
    # train_sub_verb()
    # train_sub_adj()
    # train_sub_adv()
    # train_sub_idf_top()
    # train_sub_idf_bottom()

    # train_is_early_all()
    # train_is_early_noun()
    # train_is_early_adj()
    # train_is_early_noun_adj()
    # train_is_early_verb()
    # train_is_early_adv()
    # train_is_early_verb_adv()
    # train_early_not_noun()
    # train_early_not_verb()
    # train_early_not_adj()
    # train_early_not_adv()
    # train_early_not_verb_adv()
    # train_early_not_noun_adj()

    # train_noun()
    # train_adj()
    # train_verb()
    # train_adv()
    # train_noun2()
    # train_adj2()
    # train_verb2()
    # train_adv2()

    train_is_close_all()
    train_is_adj_noun_linear_pairs()
    train_is_adv_verb_linear_pairs()

    train_bigrams()
    train_is_adj_noun_bigrams()
    train_is_adv_verb_bigrams()

    print("Done")

# _________________________________ Main for Sandbox ______________________________________
if __name__ == "__main0__":
    qry = "the crystalline lens in vertebrates, including humans"
    text = nltk.word_tokenize(qry)
    tagged = nltk.pos_tag(text)
    print(tagged)

    wn = WordNet()
    ps = PorterStemmer()
    for word in text:
        print(word + ' --> ' + ps.stem(word))

        list = wn.get_sim_terms_rw(word)
        print(list)
        print(wn.stem(ps, word, list))
        print()

    sum = 0
    for a in [('man', 0.197934595524957), ('men', 0.15318416523235803), ('world', 0.0981067125645439), ('homo', 0.08605851979345956), ('humankind', 0.06540447504302928), ('mankind', 0.06368330464716007), ('live', 0.046471600688468166), ('earth', 0.03614457831325302), ('erect', 0.030981067125645446), ('extinct', 0.024096385542168676), ('carriag', 0.02237521514629949), ('hominida', 0.020654044750430298), ('articul', 0.020654044750430298), ('speech', 0.020654044750430298), ('intellig', 0.018932874354561105), ('famili', 0.017211703958691912), ('lover', 0.015490533562822723), ('use', 0.015490533562822723), ('superior', 0.015490533562822723), ('member', 0.01376936316695353), ('alway', 0.010327022375215149), ('slight', 0.006884681583476765)]:
        sum += a[1]
    print(sum)


# _________________________________ Main Old ______________________________________
if __name__ == "__main0__":
    docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.all.1400", VectorType.DOCUMENTS, stemming_on=True)
    qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.qry", VectorType.QUERIES, stemming_on=True)
    # map {Query Ids : [Relevant Doc Ids]}
    relevant_docs = score_fs.read_human_judgement("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cranqrel", 1, 3)

    # Cosine Test
    # docs.normalize(docs)
    # qrys.normalize(docs)

    query_limit = 225
    doc_limit = 20

    # map {Query id : [Doc Ids]} where Doc Ids are sorted in order of relevance
    # cosine_results = qrys.find_closest_docs(docs, CosineFunction(docs), doc_limit=doc_limit, query_limit=query_limit)
    # cosine_avg_map = score_fs.compute_avg_map(cosine_results, relevant_docs, query_limit=query_limit)
    # print("\nCosine MAP=" + str(cosine_avg_map))

    # okapi_func = OkapiFunction(docs)
    # okapi_results = qrys.find_closest_docs(docs, okapi_func, doc_limit=doc_limit, query_limit=query_limit)
    # okapi_avg_map = score_fs.compute_avg_map(okapi_results, relevant_docs)
    # print("\nOkapi MAP=" + str(okapi_avg_map))

    # with open('out/human_judgement.json', 'w') as f1:
    #     f1.write(json.dumps(dict(relevant_docs)))
    # with open('out/cosine_results.json', 'w') as f2:
    #     f2.write(json.dumps(cosine_results))
    # with open('out/okapi_results.json', 'w') as f3:
    #     f3.write(json.dumps(okapi_results))

    #________________________________________________________________________________________________
    # okapi_func = OkapiModFunction(docs, is_early_noun_adj=True)
    # okapi_mod_results = qrys.find_closest_docs(docs, okapi_func, doc_limit=doc_limit, query_limit=query_limit)
    # okapi_mod_avg_map = score_fs.compute_avg_map(okapi_mod_results, relevant_docs)
    # print("\nOkapi Mod MAP=" + str(okapi_mod_avg_map))
    # with open('out/adi/okapi_isearlynounadj_results.json', 'w') as f3:
    #     f3.write(json.dumps(okapi_mod_results))

    print('adv verb pairs')
    influence = 1.6
    best_influence = 0
    best_map = 0
    while influence <= 2.0:
        sys.stdout.write('Influence=' + str(influence) + ' ')
        sys.stdout.flush()
        okapi_func = OkapiModFunction(docs, is_adj_noun_2gram=True, adj_noun_2gram_influence=influence)
        okapi_mod_results = qrys.find_closest_docs(docs, okapi_func, doc_limit=doc_limit, query_limit=query_limit)
        okapi_mod_avg_map = score_fs.compute_avg_map(okapi_mod_results, relevant_docs)
        if okapi_mod_avg_map > best_map:
            best_map = okapi_mod_avg_map
            best_influence = influence
        print(okapi_mod_avg_map)
        influence += 0.2
    print("Best influence = " + str(best_influence))
    print("Best MAP = " + str(best_map))


# _________________________________ Main for Graphing P+R Curves ______________________________________
if __name__ == "__main0__":
    score_fs.graph_precision_recall(225)

# _________________________________ Main for Evaluating Individual Results ______________________________________
if __name__ == "__main0__":
    # Determine the maximum number of documents/score my system can return without word substitutions
    docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.all.1400", VectorType.DOCUMENTS, stemming_on=True)
    qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.qry", VectorType.QUERIES, stemming_on=True)
    # map {Query Ids : [Relevant Doc Ids]}
    relevant_docs = score_fs.read_human_judgement("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cranqrel", 1, 3)
    # Load my system
    with open('out/cran/okapi_isearlynounadj_results.json') as f:
        results = json.load(f)

    q_id = 1
    qry = qrys.id_to_textvector[q_id]
    print('Query=' + str(qry.raw_text))
    print('Terms='+str(list(zip(qry.terms, qry.terms_pos))))
    print('Returned Doc ids=' + str(results[str(q_id)]))

    print('\nCorrect Relevant:')
    # For each relevant document, which terms from the query does the document contain?
    for doc_id in relevant_docs[q_id]:
        if doc_id in results[str(q_id)]:
            term_intersect = list(set(docs.id_to_textvector[doc_id].terms) & set(qrys.id_to_textvector[q_id].terms))
            idf = []
            for term in term_intersect:
                idf.append(round(compute_idf(docs, term), 2))
            print('Doc id:' + str(doc_id) + ', len=' + str(len(term_intersect)) + ', terms = ' + str(term_intersect) + ', ' + str(idf))

    print('\nMissed Relevant:')
    for doc_id in relevant_docs[q_id]:
        if doc_id not in results[str(q_id)]:
            term_intersect = list(set(docs.id_to_textvector[doc_id].terms) & set(qrys.id_to_textvector[q_id].terms))
            idf = []
            for term in term_intersect:
                idf.append(round(compute_idf(docs, term), 2))
            print('Doc id:' + str(doc_id) + ', len=' + str(len(term_intersect)) + ', terms = ' + str(term_intersect) + ', ' + str(idf))

    print('\nFalse Positives:')
    for doc_id in results[str(q_id)]:
        doc_id = int(doc_id)
        if doc_id not in relevant_docs[q_id] and doc_id in results[str(q_id)]:
            term_intersect = list(set(docs.id_to_textvector[doc_id].terms) & set(qrys.id_to_textvector[q_id].terms))
            idf = []
            for term in term_intersect:
                idf.append(round(compute_idf(docs, term), 2))
            print('Doc id:' + str(doc_id) + ', len=' + str(len(term_intersect)) + ', terms = ' + str(term_intersect) + ', ' + str(idf))


# _________________________________ Main for Obtaining Doc and Query Metadata ______________________________________
def get_vector_metadata(vc: VectorCollection):
    vectors = terms = nouns = adjs = verbs = advs = other = 0
    for tvector in vc.id_to_textvector.values():
        vectors += 1
        for pos in tvector.terms_pos:
            terms += 1
            if is_noun(pos):
                nouns += 1
            elif is_adjective(pos):
                adjs += 1
            elif is_verb(pos):
                verbs += 1
            elif is_adverb(pos):
                advs += 1
            else:
                other += 1
    avg_terms = terms / vectors
    avg_nouns = nouns / vectors
    avg_adjs = adjs / vectors
    avg_verbs = verbs / vectors
    avg_advs = advs / vectors
    avg_other = other / vectors if other > 0 else 0
    return (avg_terms, avg_nouns, avg_adjs, avg_verbs, avg_advs, avg_other)

if __name__ == "__main0__":
    abs_path = "/Users/Eric/Desktop/Thesis/projects/datasets"
    rel_path = "../datasets"
    f_path = abs_path
    docs, qrys, relevant_docs, dir = read_cran(f_path, title=False)
    # docs, qrys, relevant_docs, dir = read_adi(f_path)
    # docs, qrys, relevant_docs, dir = read_med(f_path)
    # docs, qrys, relevant_docs, dir = read_time(f_path)

    out_dir = "out/" + dir
    avg_terms, avg_nouns, avg_adjs, avg_verbs, avg_advs, avg_other = get_vector_metadata(qrys)
    print("Queries:")
    print("Terms="+str(avg_terms) + ", Nouns="+str(avg_nouns) + ", Adjs="+str(avg_adjs) +
          ", Verbs="+str(avg_verbs) + ", Advs="+str(avg_advs) + ", Other="+str(avg_other))
    print("Documents:")
    avg_terms, avg_nouns, avg_adjs, avg_verbs, avg_advs, avg_other = get_vector_metadata(docs)
    print("Terms="+str(avg_terms) + ", Nouns="+str(avg_nouns) + ", Adjs="+str(avg_adjs) +
          ", Verbs="+str(avg_verbs) + ", Advs="+str(avg_advs) + ", Other="+str(avg_other))




