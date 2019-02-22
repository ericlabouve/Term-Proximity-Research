# Experiments and the location of main
# Eric LaBouve (elabouve@calpoly.edu)

import ScoringFunctions as score_fs
import nltk, json, sys
import numpy as np
from VectorCollection import VectorCollection, VectorType
from DistanceFunctions import CosineFunction, OkapiFunction, OkapiModFunction, compute_idf
from WordNet import WordNet, is_noun, is_verb, is_adjective, is_adverb
from nltk.stem import PorterStemmer
from multiprocessing import Process, Queue, Array, Manager
from pathlib import Path


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

def read_cran(path, title=True, short=False):
    if title:
        docs = VectorCollection(path + "/cran/cran.all.1400", VectorType.DOCUMENTS, stemming_on=True)
    else:
        docs = VectorCollection(path + "/cran/cran.notitle.all.1400", VectorType.DOCUMENTS, stemming_on=True)
    if short:
        qrys = VectorCollection(path + "/cran/cran_short.qry", VectorType.QUERIES, stemming_on=True)
    else:
        qrys = VectorCollection(path + "/cran/cran.qry", VectorType.QUERIES, stemming_on=True)
    relevant_docs = score_fs.read_human_judgement(path + "/cran/cranqrel", 1, 3)
    return docs, qrys, relevant_docs, "cran/"


def read_adi(path, short=False):
    docs = VectorCollection(path + "/adi/ADI.ALL", VectorType.DOCUMENTS, stemming_on=True)
    if short:
        qrys = VectorCollection(path + "/adi/ADI_short.QRY", VectorType.QUERIES, stemming_on=True)
    else:
        qrys = VectorCollection(path + "/adi/ADI.QRY", VectorType.QUERIES, stemming_on=True)
    relevant_docs = score_fs.read_human_judgement(path + "/adi/ADI.REL", 0, 0)  # DIFFERENT FORMAT
    return docs, qrys, relevant_docs, "adi/"


def read_med(path, short=False):
    docs = VectorCollection(path + "/med/MED.ALL", VectorType.DOCUMENTS, stemming_on=True)
    if short:
        qrys = VectorCollection(path + "/med/MED_short.QRY", VectorType.QUERIES, stemming_on=True)
    else:
        qrys = VectorCollection(path + "/med/MED.QRY", VectorType.QUERIES, stemming_on=True)
    relevant_docs = score_fs.read_human_judgement_MED(path + "/med/MED.REL", 1, 1)  # DIFFERENT FORMAT
    return docs, qrys, relevant_docs, "med/"


def read_time(path, short=False):
    docs = VectorCollection(path + "/time/TIME_clean.ALL", VectorType.DOCUMENTS, stemming_on=True)
    if short:
        qrys = VectorCollection(path + "/time/TIME_clean_short.QUE", VectorType.QUERIES, stemming_on=True)
    else:
        qrys = VectorCollection(path + "/time/TIME_clean.QUE", VectorType.QUERIES, stemming_on=True)
    relevant_docs = score_fs.read_human_judgement_TIME(path + "/time/TIME_clean.REL")
    return docs, qrys, relevant_docs, "time/"


def read_lisa(path, title=False):
    if title:
        docs = VectorCollection(path + "/lisa/lisa_clean.all", VectorType.DOCUMENTS, stemming_on=True)
    else:
        docs = VectorCollection(path + "/lisa/lisa_clean_notitle.all", VectorType.DOCUMENTS, stemming_on=True)
    qrys = VectorCollection(path + "/lisa/LISA.QUE", VectorType.QUERIES, stemming_on=True)
    relevant_docs = score_fs.read_human_judgement_TIME(path + "/lisa/LISARJ.NUM")
    return docs, qrys, relevant_docs, "lisa/"


def read_npl(path):
    docs = VectorCollection(path + "/npl/doc-text", VectorType.DOCUMENTS, stemming_on=True)
    qrys = VectorCollection(path + "/npl/query-text_clean", VectorType.QUERIES, stemming_on=True)
    relevant_docs = score_fs.read_human_judgement_TIME(path + "/npl/rlv-ass")
    return docs, qrys, relevant_docs, "npl/"


# __________________________ Tests _______________________________

# ______________________ Round 3 ______________________
# ___________ Two Themes ___________
def test_r3_ID1():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_early_adv=True, early_adv_i=1.4)
    run_save(okapi_func, "r3_ID1")

def test_r3_ID2():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_early_noun=True, early_noun_i=1.6, is_early_adj=True, early_adj_i=1.4)
    run_save(okapi_func, "r3_ID2")

def test_r3_ID3():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_early_verb=True, early_verb_i=1.2, is_early_adv=True, early_adv_i=1.4)
    run_save(okapi_func, "r3_ID3")

def test_r3_ID4():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75)
    run_save(okapi_func, "r3_ID4")

def test_r3_ID5():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID5")

def test_r3_ID6():
    okapi_func = OkapiModFunction(docs, is_early_adv=True, early_adv_i=1.4, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75)
    run_save(okapi_func, "r3_ID6")

def test_r3_ID7():
    okapi_func = OkapiModFunction(docs, is_early_adv=True, early_adv_i=1.4, is_early_noun=True, early_noun_i=1.6, is_early_adj=True, early_adj_i=1.4, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75)
    run_save(okapi_func, "r3_ID7")

def test_r3_ID8():
    okapi_func = OkapiModFunction(docs, is_early_verb=True, early_verb_i=1.2, is_early_adv=True, early_adv_i=1.4, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75)
    run_save(okapi_func, "r3_ID8")

def test_r3_ID9():
    okapi_func = OkapiModFunction(docs, is_early_verb=True, early_verb_i=1.2, is_early_adv=True, early_adv_i=1.4, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID9")

def test_r3_ID10():
    okapi_func = OkapiModFunction(docs, is_early_adv=True, early_adv_i=1.4, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID10")

def test_r3_ID11():
    okapi_func = OkapiModFunction(docs, is_early_noun=True, early_noun_i=1.6, is_early_adj=True, early_adj_i=1.4, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID11")

def test_r3_ID12():
    okapi_func = OkapiModFunction(docs, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID12")


    # ___________ Three Themes ___________
def test_r3_ID13():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_early_adv=True, early_adv_i=1.4, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75)
    run_save(okapi_func, "r3_ID13")

def test_r3_ID14():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_early_noun=True, early_noun_i=1.6, is_early_adj=True, early_adj_i=1.4, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75)
    run_save(okapi_func, "r3_ID14")

def test_r3_ID15():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_early_verb=True, early_verb_i=1.2, is_early_adv=True, early_adv_i=1.4, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75)
    run_save(okapi_func, "r3_ID15")

def test_r3_ID16():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID16")

def test_r3_ID17():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_early_adv=True, early_adv_i=1.4, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID17")

def test_r3_ID18():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_early_noun=True, early_noun_i=1.6, is_early_adj=True, early_adj_i=1.4, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID18")

def test_r3_ID19():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_early_verb=True, early_verb_i=1.2, is_early_adv=True, early_adv_i=1.4, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID19")

def test_r3_ID20():
    okapi_func = OkapiModFunction(docs, is_early_adv=True, early_adv_i=1.4, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID20")

def test_r3_ID21():
    okapi_func = OkapiModFunction(docs, is_early_noun=True, early_noun_i=1.6, is_early_adj=True, early_adj_i=1.4, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID21")

def test_r3_ID22():
    okapi_func = OkapiModFunction(docs, is_early_verb=True, early_verb_i=1.2, is_early_adv=True, early_adv_i=1.4, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID22")


    # ___________ Four Themes ___________
def test_r3_ID23():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_early_adv=True, early_adv_i=1.4, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID23")

def test_r3_ID24():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_early_noun=True, early_noun_i=1.6, is_early_adj=True, early_adj_i=1.4, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID24")

def test_r3_ID25():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5, is_early_verb=True, early_verb_i=1.2, is_early_adv=True, early_adv_i=1.4, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r3_ID25")


# ______________________ Round 2 ______________________

# ___________ Substitutions ___________
def test_r2_sub_allallall():
    okapi_func = OkapiModFunction(docs, is_sub_all=True, sub_prob=0.1, is_sub_api_all=True,
                                  sub_api_dir=dir.replace('/', ''), is_w2v_sub_all=True, w2v_sub_sim=0.5)
    run_save(okapi_func, "r2_sub_allallall")


def test_r2_sub_bIDFbIDFbIDF():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5,
                                  is_sub_api_idf_bottom=True, sub_api_dir=dir.replace('/', ''),
                                  is_w2v_sub_idf_bottom=True, w2v_sub_sim=0.5)
    run_save(okapi_func, "r2_sub_bIDFbIDFbIDF")


def test_r2_sub_nounnounnoun():
    okapi_func = OkapiModFunction(docs, is_sub_noun=True, sub_prob=0.02, is_sub_api_noun=True,
                                  sub_api_dir=dir.replace('/', ''), is_w2v_sub_noun=True, w2v_sub_sim=0.5)
    run_save(okapi_func, "r2_sub_nounnounnoun")


def test_r2_sub_nounnounverb():
    okapi_func = OkapiModFunction(docs, is_sub_noun=True, sub_prob=0.02, is_sub_api_noun=True,
                                  sub_api_dir=dir.replace('/', ''), is_w2v_sub_verb=True, w2v_sub_sim=0.5)
    run_save(okapi_func, "r2_sub_nounnounverb")


# ___________ Term - Document ___________
def test_r2_termdoc_nounadjverbadv():
    okapi_func = OkapiModFunction(docs, is_early_noun=True, early_noun_i=1.6, is_early_verb=True, early_verb_i=1.2,
                                  is_early_adj=True, early_adj_i=1.4, is_early_adv=True, early_adv_i=1.4)
    run_save(okapi_func, "r2_termdoc_nounadjverbadv")


def test_r2_termdoc_nounverb():
    okapi_func = OkapiModFunction(docs, is_early_noun=True, early_noun_i=1.6, is_early_verb=True, early_verb_i=1.2)
    run_save(okapi_func, "r2_termdoc_nounverb")


def test_r2_termdoc_nounadj():
    okapi_func = OkapiModFunction(docs, is_early_noun=True, early_noun_i=1.6, is_early_adj=True, early_adj_i=1.4)
    run_save(okapi_func, "r2_termdoc_nounadj")


def test_r2_termdoc_verbadv():
    okapi_func = OkapiModFunction(docs, is_early_verb=True, early_verb_i=1.2, is_early_adv=True, early_adv_i=1.4)
    run_save(okapi_func, "r2_termdoc_verbadv")


# ___________ Parts of Speech ___________
def test_r2_pos_noun16adj12verb12adv04():
    okapi_func = OkapiModFunction(docs, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2, is_verb=True,
                                  verb_influence=1.2, is_adv=True, adv_influence=0.4)
    run_save(okapi_func, "r2_pos_noun16adj12verb12adv04")


def test_r2_pos_noun16adj08verb12adv04():
    okapi_func = OkapiModFunction(docs, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=0.8, is_verb=True,
                                  verb_influence=1.2, is_adv=True, adv_influence=0.4)
    run_save(okapi_func, "r2_pos_noun16adj08verb12adv04")


def test_r2_pos_noun16adj12verb06adv04():
    okapi_func = OkapiModFunction(docs, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2, is_verb=True,
                                  verb_influence=0.6, is_adv=True, adv_influence=0.4)
    run_save(okapi_func, "r2_pos_noun16adj12verb06adv04")


def test_r2_pos_noun16adj08verb06adv04():
    okapi_func = OkapiModFunction(docs, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=0.8, is_verb=True,
                                  verb_influence=0.6, is_adv=True, adv_influence=0.4)
    run_save(okapi_func, "r2_pos_noun16adj08verb06adv04")


def test_r2_pos_noun16verb12():
    okapi_func = OkapiModFunction(docs, is_noun=True, noun_influence=1.6, is_verb=True, verb_influence=1.2)
    run_save(okapi_func, "r2_pos_noun16verb12")


def test_r2_pos_noun16verb06():
    okapi_func = OkapiModFunction(docs, is_noun=True, noun_influence=1.6, is_verb=True, verb_influence=0.6)
    run_save(okapi_func, "r2_pos_noun16verb06")


def test_r2_pos_noun16adj12():
    okapi_func = OkapiModFunction(docs, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "r2_pos_noun16adj12")


def test_r2_pos_noun16adj08():
    okapi_func = OkapiModFunction(docs, is_noun=True, noun_influence=1.6, is_adj=True, adj_influence=0.8)
    run_save(okapi_func, "r2_pos_noun16adj08")


def test_r2_pos_verb12adv04():
    okapi_func = OkapiModFunction(docs, is_verb=True, verb_influence=1.2, is_adv=True, adv_influence=0.4)
    run_save(okapi_func, "r2_pos_verb12adv04")


def test_r2_pos_verb06adv04():
    okapi_func = OkapiModFunction(docs, is_verb=True, verb_influence=0.6, is_adv=True, adv_influence=0.4)
    run_save(okapi_func, "r2_pos_verb06adv04")


# ___________ Term - Term ___________
def test_r2_termterm_cpAdjNounBAll():
    okapi_func = OkapiModFunction(docs, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75,
                                  is_bigram=True, bigram_influence=1.2)
    run_save(okapi_func, "r2_termterm_cpAdjNounBAll")


def test_r2_termterm_cpAdjNounBAdjNoun():
    okapi_func = OkapiModFunction(docs, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75,
                                  is_adj_noun_2gram=True, adj_noun_2gram_influence=1.2)
    run_save(okapi_func, "r2_termterm_cpAdjNounBAdjNoun")


# ______________________ Round 1 ______________________

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
    okapi_func = OkapiModFunction(docs, is_sub_all=True, sub_prob=0.1)
    run_save(okapi_func, "sub all i=.1")


def test_sub_noun():
    okapi_func = OkapiModFunction(docs, is_sub_noun=True, sub_prob=0.02)
    run_save(okapi_func, "sub noun i=.02")


def test_sub_verb():
    okapi_func = OkapiModFunction(docs, is_sub_verb=True, sub_prob=0.06)
    run_save(okapi_func, "sub verb i=.06")


def test_sub_adj():
    okapi_func = OkapiModFunction(docs, is_sub_adj=True, sub_prob=0.1)
    run_save(okapi_func, "sub adj i=.1")


def test_sub_adv():
    okapi_func = OkapiModFunction(docs, is_sub_adv=True, sub_prob=0.04)
    run_save(okapi_func, "sub adv i=.04")


def test_sub_idf_top():
    okapi_func = OkapiModFunction(docs, is_sub_idf_top=True, sub_prob=0.1, sub_idf_top=5)
    run_save(okapi_func, "sub idf top=5 i=.1")


def test_sub_idf_bottom():
    okapi_func = OkapiModFunction(docs, is_sub_idf_bottom=True, sub_prob=0.1, sub_idf_bottom=5)
    run_save(okapi_func, "sub idf bottom=5 i=.1")


def test_sub_api_all():
    okapi_func = OkapiModFunction(docs, is_sub_api_all=True, sub_api_dir=dir.replace('/', ''))
    run_save(okapi_func, "sub api all i=.9")


def test_sub_api_noun():
    okapi_func = OkapiModFunction(docs, is_sub_api_noun=True, sub_api_dir=dir.replace('/', ''))
    run_save(okapi_func, "sub api noun i=.9")


def test_sub_api_verb():
    okapi_func = OkapiModFunction(docs, is_sub_api_verb=True, sub_api_dir=dir.replace('/', ''))
    run_save(okapi_func, "sub api verb i=.9")


def test_sub_api_adj():
    okapi_func = OkapiModFunction(docs, is_sub_api_adj=True, sub_api_dir=dir.replace('/', ''))
    run_save(okapi_func, "sub api adj i=.9")


def test_sub_api_adv():
    okapi_func = OkapiModFunction(docs, is_sub_api_adv=True, sub_api_dir=dir.replace('/', ''))
    run_save(okapi_func, "sub api adv i=.9")


def test_sub_api_idf_top():
    okapi_func = OkapiModFunction(docs, is_sub_api_idf_top=True, sub_api_dir=dir.replace('/', ''))
    run_save(okapi_func, "sub api idf top=5 i=.9")


def test_sub_api_idf_bottom():
    okapi_func = OkapiModFunction(docs, is_sub_api_idf_bottom=True, sub_api_dir=dir.replace('/', ''))
    run_save(okapi_func, "sub api idf bottom=5 i=.9")


def test_w2v_sub_all():
    okapi_func = OkapiModFunction(docs, is_w2v_sub_all=True, w2v_sub_sim=0.5)  # 1 std dev bellow mean
    run_save(okapi_func, "w2v_sub all i=0.5")


def test_w2v_sub_noun():
    okapi_func = OkapiModFunction(docs, is_w2v_sub_noun=True, w2v_sub_sim=0.5)
    run_save(okapi_func, "w2v_sub noun i=0.5")


def test_w2v_sub_verb():
    okapi_func = OkapiModFunction(docs, is_w2v_sub_verb=True, w2v_sub_sim=0.5)
    run_save(okapi_func, "w2v_sub verb i=0.5")


def test_w2v_sub_adj():
    okapi_func = OkapiModFunction(docs, is_w2v_sub_adj=True, w2v_sub_sim=0.5)
    run_save(okapi_func, "w2v_sub adj i=0.5")


def test_w2v_sub_adv():
    okapi_func = OkapiModFunction(docs, is_w2v_sub_adv=True, w2v_sub_sim=0.5)
    run_save(okapi_func, "w2v_sub adv i=0.5")


def test_w2v_sub_idf_top():
    okapi_func = OkapiModFunction(docs, is_w2v_sub_idf_top=True, w2v_sub_sim=0.5, sub_idf_top=5)
    run_save(okapi_func, "w2v_sub idf top=5 i=0.5")


def test_w2v_sub_idf_bottom():
    okapi_func = OkapiModFunction(docs, is_w2v_sub_idf_bottom=True, w2v_sub_sim=0.5, sub_idf_bottom=5)
    run_save(okapi_func, "w2v_sub idf bottom=5 i=0.5")


def test_early_all():
    okapi_func = OkapiModFunction(docs, is_early=True, early_term_influence=2.2)
    run_save(okapi_func, "early all i=2.2")


def test_early_noun():
    okapi_func = OkapiModFunction(docs, is_early_noun=True, early_noun_i=1.6)
    run_save(okapi_func, "early noun i=1.6")


def test_early_verb():
    okapi_func = OkapiModFunction(docs, is_early_verb=True, early_verb_i=1.2)
    run_save(okapi_func, "early verb i=1.2")


def test_early_adj():
    okapi_func = OkapiModFunction(docs, is_early_adj=True, early_adj_i=1.4)
    run_save(okapi_func, "early adj i=1.4")


def test_early_adv():
    okapi_func = OkapiModFunction(docs, is_early_adv=True, early_adv_i=1.4)
    run_save(okapi_func, "early adv i=1.4")


def test_early_noun_adj():
    okapi_func = OkapiModFunction(docs, is_early_noun_adj=True, early_term_influence=2.6)
    run_save(okapi_func, "early noun adj i=2.6")


def test_early_verb_adv():
    okapi_func = OkapiModFunction(docs, is_early_verb_adv=True, early_term_influence=1.2)
    run_save(okapi_func, "early verb adv i=1.2")


def test_early_not_noun():
    okapi_func = OkapiModFunction(docs, is_early_not_noun=True, early_term_influence=1.4)
    run_save(okapi_func, "early not noun i=1.4")


def test_early_not_verb():
    okapi_func = OkapiModFunction(docs, is_early_not_verb=True, early_term_influence=2.6)
    run_save(okapi_func, "early not verb i=2.6")


def test_early_not_adj():
    okapi_func = OkapiModFunction(docs, is_early_not_adj=True, early_term_influence=1.6)
    run_save(okapi_func, "early not adj i=1.6")


def test_early_not_adv():
    okapi_func = OkapiModFunction(docs, is_early_not_adv=True, early_term_influence=2.2)
    run_save(okapi_func, "early not adv i=2.2")


def test_early_not_verb_adv():
    okapi_func = OkapiModFunction(docs, is_early_not_verb_adv=True, early_term_influence=2.6)
    run_save(okapi_func, "early not verb adv i=2.6")


def test_early_not_noun_adj():
    okapi_func = OkapiModFunction(docs, is_early_not_noun_adj=True, early_term_influence=1.2)
    run_save(okapi_func, "early not noun adj i=1.2")


def test_noun():
    okapi_func = OkapiModFunction(docs, is_noun=True, noun_influence=1.6)
    run_save(okapi_func, "noun i=1.6")


def test_adj():
    okapi_func = OkapiModFunction(docs, is_adj=True, adj_influence=1.2)
    run_save(okapi_func, "adj i=1.2")


def test_verb():
    okapi_func = OkapiModFunction(docs, is_verb=True, verb_influence=1.2)
    run_save(okapi_func, "verb i=1.2")


def test_adv():
    okapi_func = OkapiModFunction(docs, is_adv=True, adv_influence=1.6)
    run_save(okapi_func, "adv i=1.6")


def test_noun2():
    okapi_func = OkapiModFunction(docs, is_noun=True, noun_influence=0.8)
    run_save(okapi_func, "noun i=0.8")


def test_adj2():
    okapi_func = OkapiModFunction(docs, is_adj=True, adj_influence=0.8)
    run_save(okapi_func, "adj i=0.8")


def test_verb2():
    okapi_func = OkapiModFunction(docs, is_verb=True, verb_influence=0.6)
    run_save(okapi_func, "verb i=0.6")


def test_adv2():
    okapi_func = OkapiModFunction(docs, is_adv=True, adv_influence=0.4)
    run_save(okapi_func, "adv i=0.4")


def test_is_close_all():
    okapi_func = OkapiModFunction(docs, is_close_pairs=True, close_pairs_m=-.25, close_pairs_b=1.75)
    run_save(okapi_func, "close_pairs i=1.75")


def test_is_adj_noun_linear_pairs():
    okapi_func = OkapiModFunction(docs, is_adj_noun_linear_pairs=True, adj_noun_pairs_m=-.25, adj_noun_pairs_b=1.75)
    run_save(okapi_func, "adj_noun_linear_pairs i=1.75")


def test_is_adv_verb_linear_pairs():
    okapi_func = OkapiModFunction(docs, is_adv_verb_linear_pairs=True, adv_verb_pairs_m=-.25, adv_verb_pairs_b=1.75)
    run_save(okapi_func, "adv_verb_linear_pairs i=1.75")


def test_bigrams():
    okapi_func = OkapiModFunction(docs, is_bigram=True, bigram_influence=1.2)
    run_save(okapi_func, "bigrams i=1.2")


def test_adj_noun_bigrams():
    okapi_func = OkapiModFunction(docs, is_adj_noun_2gram=True, adj_noun_2gram_influence=1.2)
    run_save(okapi_func, "adj_noun_bigrams i=1.2")


def test_adv_verb_bigrams():
    okapi_func = OkapiModFunction(docs, is_adv_verb_2gram=True, adv_verb_2gram_influence=2.8)
    run_save(okapi_func, "adv_verb_bigrams i=2.8")


# __________________________ Train _______________________________
# ______________________ Round 1 ______________________
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


def train_w2v_sub_all():
    influence = 0.6  # Mean value
    funcs = []
    while influence >= 0.35:
        func = OkapiModFunction(docs, is_w2v_sub_all=True, w2v_sub_sim=influence)
        funcs.append((func, "w2v_sub_all" + " i=" + str(influence)))
        influence -= 0.05  # Decrement by 1/2 standard deviation at a time
    run_funcs(funcs)


def train_w2v_sub_noun():
    influence = 0.6
    funcs = []
    while influence >= 0.35:
        func = OkapiModFunction(docs, is_w2v_sub_noun=True, w2v_sub_sim=influence)
        funcs.append((func, "w2v_sub_noun" + " i=" + str(influence)))
        influence -= 0.05
    run_funcs(funcs)


def train_w2v_sub_verb():
    influence = 0.6
    funcs = []
    while influence >= 0.35:
        func = OkapiModFunction(docs, is_w2v_sub_verb=True, w2v_sub_sim=influence)
        funcs.append((func, "w2v_sub_verb" + " i=" + str(influence)))
        influence -= 0.05
    run_funcs(funcs)


def train_w2v_sub_adj():
    influence = 0.6
    funcs = []
    while influence >= 0.35:
        func = OkapiModFunction(docs, is_w2v_sub_adj=True, w2v_sub_sim=influence)
        funcs.append((func, "w2v_sub_adj" + " i=" + str(influence)))
        influence -= 0.05
    run_funcs(funcs)


def train_w2v_sub_adv():
    influence = 0.6
    funcs = []
    while influence >= 0.35:
        func = OkapiModFunction(docs, is_w2v_sub_adv=True, w2v_sub_sim=influence)
        funcs.append((func, "w2v_sub_adv" + " i=" + str(influence)))
        influence -= 0.05
    run_funcs(funcs)


def train_w2v_sub_idf_top():
    influence = 0.6
    funcs = []
    while influence >= 0.35:
        func = OkapiModFunction(docs, is_w2v_sub_idf_top=True, w2v_sub_sim=influence, w2v_sub_idf_top=5)
        funcs.append((func, "w2v_sub_idf top=5" + " i=" + str(influence)))
        influence -= 0.05
    run_funcs(funcs)


def train_w2v_sub_idf_bottom():
    influence = 0.6
    funcs = []
    while influence >= 0.35:
        func = OkapiModFunction(docs, is_w2v_sub_idf_bottom=True, w2v_sub_sim=influence, w2v_sub_idf_bottom=5)
        funcs.append((func, "w2v_sub_idf bottom=5" + " i=" + str(influence)))
        influence -= 0.05
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
if __name__ == "__main0__":
    # abs_path = "/Users/Eric/Desktop/Thesis/projects/datasets"
    abs_path = r"C:\Users\Eric\Documents\Thesis\src\Term-Proximity-Research\datasets"
    # rel_path = "../datasets"
    rel_path = r"..\datasets"
    f_path = abs_path
    # docs, qrys, relevant_docs, dir = read_cran(f_path, title=False)
    # docs, qrys, relevant_docs, dir = read_adi(f_path)
    # docs, qrys, relevant_docs, dir = read_med(f_path)
    # docs, qrys, relevant_docs, dir = read_time(f_path)
    # print('No Title')
    docs, qrys, relevant_docs, dir = read_lisa(f_path, title=False)
    # docs, qrys, relevant_docs, dir = read_npl(f_path)

    query_limit = -1  # Use all queries
    doc_limit = -1  # Use all documents
    # out_dir = "out/train_cran/" + dir
    out_dir = "out\\train_cran\\" + dir.replace("/", "\\")

    # with open(out_dir + 'human_judgement.json', 'w') as f1:
    #     f1.write(json.dumps(dict(relevant_docs)))

    # ______________________ Round 3 ______________________
    # ___________ Two Themes ___________
    # test_r3_ID1()
    # test_r3_ID2()
    # test_r3_ID3()
    # test_r3_ID4()
    # test_r3_ID5()
    # test_r3_ID6()
    # test_r3_ID7()
    # test_r3_ID8()
    # test_r3_ID9()
    # test_r3_ID10()
    # test_r3_ID11()
    # test_r3_ID12()

    # ___________ Three Themes ___________
    # test_r3_ID13()
    # test_r3_ID14()
    # test_r3_ID15()
    # test_r3_ID16()
    # test_r3_ID17()
    # test_r3_ID18()
    # test_r3_ID19()
    # test_r3_ID20()
    # test_r3_ID21()
    # test_r3_ID22()

    # ___________ Four Themes ___________
    # test_r3_ID23()
    # test_r3_ID24()
    # test_r3_ID25()

    # ______________________ Round 2 ______________________

    # ___________ Substitutions ___________
    # test_r2_sub_allallall()
    # test_r2_sub_bIDFbIDFbIDF()
    # test_r2_sub_nounnounnoun()
    # test_r2_sub_nounnounverb()

    # ___________ Term - Document ___________
    # test_r2_termdoc_nounadjverbadv()
    # test_r2_termdoc_nounverb()
    # test_r2_termdoc_nounadj()
    # test_r2_termdoc_verbadv()

    # ___________ Parts of Speech ___________
    # test_r2_pos_noun16adj12verb12adv04()
    # test_r2_pos_noun16adj08verb12adv04()
    # test_r2_pos_noun16adj12verb06adv04()
    # test_r2_pos_noun16adj08verb06adv04()
    # test_r2_pos_noun16verb12()
    # test_r2_pos_noun16verb06()
    # test_r2_pos_noun16adj12()
    # test_r2_pos_noun16adj08()
    # test_r2_pos_verb12adv04()
    # test_r2_pos_verb06adv04()

    # ___________ Term - Term ___________
    # test_r2_termterm_cpAdjNounBAll()
    # test_r2_termterm_cpAdjNounBAdjNoun()

    # ______________________ Round 1 ______________________

    # ___________ Tests ___________
    test_cosine()
    test_okapi()
    #
    # test_is_remove_adj()
    # test_is_remove_adv()
    #
    # test_sub_all()
    # test_sub_noun()
    # test_sub_verb()
    # test_sub_adj()
    # test_sub_adv()
    # test_sub_idf_top()
    # test_sub_idf_bottom()

    # test_sub_api_all()
    # test_sub_api_noun()
    # test_sub_api_verb()
    # test_sub_api_adj()
    # test_sub_api_adv()
    # test_sub_api_idf_top()
    # test_sub_api_idf_bottom()

    # test_w2v_sub_all()
    # test_w2v_sub_noun()
    # test_w2v_sub_verb()
    # test_w2v_sub_adj()
    # test_w2v_sub_adv()
    # test_w2v_sub_idf_top()
    # test_w2v_sub_idf_bottom()

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
    #
    # test_noun()
    # test_adj()
    # test_verb()
    # test_adv()
    # test_noun2()
    # test_adj2()
    # test_verb2()
    # test_adv2()
    #
    # test_is_close_all()
    # test_is_adj_noun_linear_pairs()
    # test_is_adv_verb_linear_pairs()
    #
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

    # train_w2v_sub_all()
    # train_w2v_sub_noun()
    # train_w2v_sub_verb()
    # train_w2v_sub_adj()
    # train_w2v_sub_adv()
    # train_w2v_sub_idf_top()
    # train_w2v_sub_idf_bottom()

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

    # train_is_close_all()
    # train_is_adj_noun_linear_pairs()
    # train_is_adv_verb_linear_pairs()

    # train_bigrams()
    # train_is_adj_noun_bigrams()
    # train_is_adv_verb_bigrams()

    print("Done")


# _________________________________ Main for Lucene ______________________________________
if __name__ == "__main0__":
    # notitle_results = r"C:\Users\Eric\Documents\Thesis\src\Term-Proximity-Research\python project\out\train_cran\lisa\lucene_notitles_results.json"
    notitle_results = r"C:\Users\Eric\Documents\Thesis\src\Term-Proximity-Research\python project\out\train_cran\lisa\lucene_notitles_noedit_results.json"
    # title_results = r"C:\Users\Eric\Documents\Thesis\src\Term-Proximity-Research\python project\out\train_cran\lisa\lucene_titles_results.json"
    title_results = r"C:\Users\Eric\Documents\Thesis\src\Term-Proximity-Research\python project\out\train_cran\lisa\lucene_titles_noedit_results.json"

    with open(notitle_results) as f:
        notitle_results_json = json.load(f)
    with open(title_results) as f:
        title_results_json = json.load(f)

    relevant_docs = score_fs.read_human_judgement_TIME(r"C:\Users\Eric\Documents\Thesis\src\Term-Proximity-Research\datasets\lisa\LISARJ.NUM")

    # Need to convert keys from string to int
    def stringKeysToInt(json):
        d = {}
        for k, v in json.items():
            d[int(k)] = v
        return d

    notitle_results_json = stringKeysToInt(notitle_results_json)
    title_results_json = stringKeysToInt(title_results_json)

    print(notitle_results_json)

    avg_map = score_fs.compute_avg_map(notitle_results_json, relevant_docs)
    print("Lucene with no titles Lisa MAP  = " + str(avg_map))
    avg_map = score_fs.compute_avg_map(title_results_json, relevant_docs)
    print("Lucene with titles Lisa MAP  = " + str(avg_map))


# _________________________________ Main for Solr ______________________________________
if __name__ == "__main__":
    notitle_results = r"C:\Users\Eric\Documents\Thesis\src\Term-Proximity-Research\python project\out\train_cran\lisa\solr_notitles_noedit_results.json"
    title_results = r"C:\Users\Eric\Documents\Thesis\src\Term-Proximity-Research\python project\out\train_cran\lisa\solr_titles_noedit_results.json"

    with open(notitle_results) as f:
        notitle_results_json = json.load(f)
    with open(title_results) as f:
        title_results_json = json.load(f)

    relevant_docs = score_fs.read_human_judgement_TIME(r"C:\Users\Eric\Documents\Thesis\src\Term-Proximity-Research\datasets\lisa\LISARJ.NUM")

    # Need to convert keys from string to int
    def stringKeysToInt(json):
        d = {}
        for k, v in json.items():
            d[int(k)] = v
        return d

    notitle_results_json = stringKeysToInt(notitle_results_json)
    title_results_json = stringKeysToInt(title_results_json)

    print(notitle_results_json)

    avg_map = score_fs.compute_avg_map(notitle_results_json, relevant_docs)
    print("Solr with no titles Lisa MAP  = " + str(avg_map))
    avg_map = score_fs.compute_avg_map(title_results_json, relevant_docs)
    print("Solr with titles Lisa MAP  = " + str(avg_map))


# _________________________________ Main for Sandbox ______________________________________
if __name__ == "__main0__":
    # Calculates the mean and standard deviation for all query term substitutions values from WordNet
    abs_path = "/Users/Eric/Desktop/Thesis/projects/datasets"
    rel_path = "../datasets"
    f_path = abs_path
    _, qrys1, _, _ = read_cran(f_path, title=False)
    _, qrys2, _, _ = read_adi(f_path)
    _, qrys3, _, _ = read_med(f_path)
    _, qrys4, _, _ = read_time(f_path)
    print('read')

    qTerms = []
    for qryCollections in [qrys1, qrys2, qrys3, qrys4]:
        for qry in qryCollections.id_to_textvector.values():
            for sub_list in qry.terms_sub:
                if len(sub_list) > 0:
                    for sub, prob in sub_list:
                        qTerms.append((sub, prob))
            print('completed one qry')
        print('completed one qryCollections')
    l = np.array(qTerms)
    l_probs = np.array([float(x[1]) for x in l])
    print(l_probs.mean())
    print(l_probs.std())

# _________________________________ Main 2 ______________________________________
if __name__ == "__main0__":
    docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.all.1400", VectorType.DOCUMENTS,
                            stemming_on=True)
    qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.qry", VectorType.QUERIES,
                            stemming_on=True)
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
    #
    # okapi_func = OkapiFunction(docs)
    # okapi_results = qrys.find_closest_docs(docs, okapi_func, doc_limit=doc_limit, query_limit=query_limit)
    # okapi_avg_map = score_fs.compute_avg_map(okapi_results, relevant_docs)
    # print("\nOkapi MAP=" + str(okapi_avg_map))

    with open('out/human_judgement.json', 'w') as f1:
        f1.write(json.dumps(dict(relevant_docs)))
    # with open('out/cosine_results.json', 'w') as f2:
    #     f2.write(json.dumps(cosine_results))
    # with open('out/okapi_results.json', 'w') as f3:
    #     f3.write(json.dumps(okapi_results))

    # ________________________________________________________________________________________________
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

# _________________________________ Main for Graphing P+R Curves and Recall data  ______________________________________
if __name__ == "__main0__":
    score_fs.graph_precision_recall(35)

if __name__ == "__main0__":
    # dir = 'out/train_cran/lisa/'
    dir = 'out\\train_cran\\lisa\\'
    # Calculates all the recall scores at the specified document intervals
    score_fs.calc_all_recall(dir, [5, 10, 20])

# _________________________________ Main for Evaluating Individual Results ______________________________________
if __name__ == "__main0__":
    # Determine the maximum number of documents/score my system can return without word substitutions
    docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.all.1400", VectorType.DOCUMENTS,
                            stemming_on=True)
    qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.qry", VectorType.QUERIES,
                            stemming_on=True)
    # map {Query Ids : [Relevant Doc Ids]}
    relevant_docs = score_fs.read_human_judgement("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cranqrel", 1, 3)
    # Load my system
    with open('out/cran/okapi_isearlynounadj_results.json') as f:
        results = json.load(f)

    q_id = 1
    qry = qrys.id_to_textvector[q_id]
    print('Query=' + str(qry.raw_text))
    print('Terms=' + str(list(zip(qry.terms, qry.terms_pos))))
    print('Returned Doc ids=' + str(results[str(q_id)]))

    print('\nCorrect Relevant:')
    # For each relevant document, which terms from the query does the document contain?
    for doc_id in relevant_docs[q_id]:
        if doc_id in results[str(q_id)]:
            term_intersect = list(set(docs.id_to_textvector[doc_id].terms) & set(qrys.id_to_textvector[q_id].terms))
            idf = []
            for term in term_intersect:
                idf.append(round(compute_idf(docs, term), 2))
            print('Doc id:' + str(doc_id) + ', len=' + str(len(term_intersect)) + ', terms = ' + str(
                term_intersect) + ', ' + str(idf))

    print('\nMissed Relevant:')
    for doc_id in relevant_docs[q_id]:
        if doc_id not in results[str(q_id)]:
            term_intersect = list(set(docs.id_to_textvector[doc_id].terms) & set(qrys.id_to_textvector[q_id].terms))
            idf = []
            for term in term_intersect:
                idf.append(round(compute_idf(docs, term), 2))
            print('Doc id:' + str(doc_id) + ', len=' + str(len(term_intersect)) + ', terms = ' + str(
                term_intersect) + ', ' + str(idf))

    print('\nFalse Positives:')
    for doc_id in results[str(q_id)]:
        doc_id = int(doc_id)
        if doc_id not in relevant_docs[q_id] and doc_id in results[str(q_id)]:
            term_intersect = list(set(docs.id_to_textvector[doc_id].terms) & set(qrys.id_to_textvector[q_id].terms))
            idf = []
            for term in term_intersect:
                idf.append(round(compute_idf(docs, term), 2))
            print('Doc id:' + str(doc_id) + ', len=' + str(len(term_intersect)) + ', terms = ' + str(
                term_intersect) + ', ' + str(idf))


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


if __name__ == "_7_main__":
    # abs_path = "/Users/Eric/Desktop/Thesis/projects/datasets"
    # rel_path = "../datasets"
    abs_path = r"C:\Users\Eric\Documents\Thesis\src\Term-Proximity-Research\datasets"
    rel_path = r"..\datasets"
    f_path = abs_path
    # docs, qrys, relevant_docs, dir = read_cran(f_path, title=False, short=True)
    # docs, qrys, relevant_docs, dir = read_adi(f_path, short=True)
    # docs, qrys, relevant_docs, dir = read_med(f_path, short=True)
    # docs, qrys, relevant_docs, dir = read_time(f_path, short=True)
    print("Title")
    docs, qrys, relevant_docs, dir = read_lisa(f_path, title=True)

    out_dir = "out/" + dir
    avg_terms, avg_nouns, avg_adjs, avg_verbs, avg_advs, avg_other = get_vector_metadata(qrys)
    print("Queries:")
    print("Terms=" + str(avg_terms) + ", Nouns=" + str(avg_nouns) + ", Adjs=" + str(avg_adjs) +
          ", Verbs=" + str(avg_verbs) + ", Advs=" + str(avg_advs) + ", Other=" + str(avg_other))
    print("Documents:")
    avg_terms, avg_nouns, avg_adjs, avg_verbs, avg_advs, avg_other = get_vector_metadata(docs)
    print("Terms=" + str(avg_terms) + ", Nouns=" + str(avg_nouns) + ", Adjs=" + str(avg_adjs) +
          ", Verbs=" + str(avg_verbs) + ", Advs=" + str(avg_advs) + ", Other=" + str(avg_other))
