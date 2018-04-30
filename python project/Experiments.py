# Experiments and the location of main
# Eric LaBouve (elabouve@calpoly.edu)

import ScoringFunctions as score_fs
import nltk, json, sys
from VectorCollection import VectorCollection, VectorType
from DistanceFunctions import CosineFunction, OkapiFunction, OkapiModFunction, compute_idf
from WordNet import WordNet
from multiprocessing import Process, Queue, Array, Manager



if __name__ == "__main__":
    # qry = "what problems of heat conduction in composite slabs have been solved so far ."
    # [('what', 'WP'), ('problems', 'NNS'), ('of', 'IN'), ('heat', 'NN'), ('conduction', 'NN'), ('in', 'IN'), ('composite', 'JJ'), ('slabs', 'NNS'), ('have', 'VBP'), ('been', 'VBN'), ('solved', 'VBN'), ('so', 'RB'), ('far', 'RB'), ('.', '.')]
    # text = nltk.word_tokenize(qry)
    # print(nltk.pos_tag(text))

    total = 0
    for tup in [('resolved', 0.20200892857142858), ('solving', 0.0859375), ('solve', 0.08370535714285714), ('unresolved', 0.041294642857142856), ('cleared', 0.04017857142857143), ('debt', 0.0390625), ('clear', 0.03459821428571429), ('settle', 0.033482142857142856), ('resolving', 0.033482142857142856), ('clearing', 0.03236607142857143), ('unsolved', 0.03236607142857143), ('find', 0.027901785714285716), ('resolve', 0.024553571428571428), ('solution', 0.022321428571428572), ('licked', 0.021205357142857144), ('work', 0.018973214285714284), ('wrought', 0.015625), ('works', 0.015625), ('licking', 0.014508928571428572), ('working', 0.014508928571428572), ('lick', 0.013392857142857142), ('open-and-shut', 0.011160714285714286), ('example', 0.008928571428571428), ('x', 0.008928571428571428), ('insolvable', 0.008928571428571428), ('equation', 0.008928571428571428), ('old', 0.0078125), ('easily', 0.0078125), ('meaning', 0.0078125), ('solvable', 0.006696428571428571), ('unsoluble', 0.006696428571428571), ('obvious', 0.005580357142857143), ('understand', 0.005580357142857143), ('remain', 0.004464285714285714), ('being', 0.004464285714285714), ('unresolvable', 0.004464285714285714), ('resolvable', 0.004464285714285714), ('boss', 0.0033482142857142855), ('situation', 0.0033482142857142855), ('unpleasant', 0.0033482142857142855), ('decided', 0.0033482142857142855), ('going', 0.002232142857142857), ('capable', 0.002232142857142857), ('soluble', 0.002232142857142857), ('many', 0.002232142857142857), ('unsolvable', 0.002232142857142857), ('case', 0.002232142857142857), ('problem', 0.002232142857142857), ('math', 0.002232142857142857), ('understanding', 0.002232142857142857), ('exercise', 0.0011160714285714285), ('develop', 0.0011160714285714285), ('perfectly', 0.0011160714285714285), ('finance', 0.0011160714285714285), ('l', 0.0011160714285714285), ('task', 0.0011160714285714285)]:
        total += tup[1]
    print(total)
    #
    # wn = WordNet()
    # print(wn.get_sim_terms_rw('solved'))
    # print(wn.get_syns(('solved', 'VBN')))


def run(queue, okapi_func, label):
    okapi_mod_results = qrys.find_closest_docs(docs, okapi_func, doc_limit=doc_limit, query_limit=query_limit)
    okapi_mod_avg_map = score_fs.compute_avg_map(okapi_mod_results, relevant_docs)
    queue.append((label, okapi_mod_avg_map))

if __name__ == "__main0__":
    docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.all.1400", VectorType.DOCUMENTS, stemming_on=True)
    qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.qry", VectorType.QUERIES, stemming_on=True)
    relevant_docs = score_fs.read_human_judgement("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cranqrel", 1, 3)

    # docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/adi/ADI.ALL", VectorType.DOCUMENTS, stemming_on=True)
    # qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/adi/ADI.QRY", VectorType.QUERIES, stemming_on=True)
    # relevant_docs = score_fs.read_human_judgement("/Users/Eric/Desktop/Thesis/projects/datasets/adi/ADI.REL", 0, 0)  # DIFFERENT FORMAT

    # docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/med/MED.ALL", VectorType.DOCUMENTS, stemming_on=True)
    # qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/med/MED.QRY", VectorType.QUERIES, stemming_on=True)
    # relevant_docs = score_fs.read_human_judgement_MED("/Users/Eric/Desktop/Thesis/projects/datasets/med/MED.REL", 1, 1)  # DIFFERENT FORMAT

    # docs.normalize(docs)
    # qrys.normalize(docs)

    m = Manager()
    q = m.list()
    query_limit = -1  # Use all queries
    doc_limit = -1  # Use all documents
    process_list = []

    # Loop for running processes in parallel that differ by level of influence
    # influence = 1.4
    # while influence <= 1.6:
    #     okapi_func = OkapiModFunction(docs, is_early_noun_adj=True, is_adj_noun_linear_pairs=True, adj_noun_pairs_b=influence)
    #     p = Process(target=run, args=(q, okapi_func, 'is_early_noun_adj inf=' + str(influence)))
    #     process_list.append(p)
    #     influence += 0.1

    okapi_func1 = OkapiModFunction(docs, is_early_noun_adj=True, is_adj_noun_linear_pairs=True, adj_noun_pairs_b=1.5)
    # okapi_func2 = OkapiModFunction(docs, is_early_noun_adj=True, is_adj_noun_linear_pairs=True, adj_noun_pairs_b=1.4)
    p1 = Process(target=run, args=(q, okapi_func1, 'inf=1.5'))
    # p2 = Process(target=run, args=(q, okapi_func2, 'inf=1.4'))
    process_list.append(p1)
    # process_list.append(p2)

    # Start all processes
    for p in process_list:
        p.start()

    # Wait for all processes to end
    for p in process_list:
        p.join()

    # Sort based on base MAP score and display stats
    q = sorted(q, key=lambda x: x[1], reverse=True)
    print()
    print(q)


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


if __name__ == "__main0__":
    score_fs.graph_precision_recall(225)


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














