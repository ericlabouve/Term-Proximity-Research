# Experiments and the location of main
# Eric LaBouve (elabouve@calpoly.edu)

import ScoringFunctions as score_fs
import nltk, json, sys
from VectorCollection import VectorCollection, VectorType
from DistanceFunctions import CosineFunction, OkapiFunction, OkapiModFunction, compute_idf
from WordNet import WordNet
from nltk.stem import PorterStemmer
from multiprocessing import Process, Queue, Array, Manager



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

    # sum = 0
    # for a in [('resolv', 0.28651685393258425), ('solv', 0.18202247191011234), ('clear', 0.11910112359550562), ('unsolv', 0.04494382022471911), ('work', 0.043820224719101124), ('debt', 0.04157303370786517), ('unresolv', 0.03707865168539326), ('lick', 0.03595505617977528), ('solut', 0.02247191011235955), ('find', 0.017977528089887642), ('settl', 0.016853932584269662), ('wrought', 0.015730337078651686), ('equat', 0.011235955056179775), ('x', 0.008988764044943821), ('unsolubl', 0.008988764044943821), ('open-and-shut', 0.008988764044943821), ('boss', 0.007865168539325843), ('old', 0.006741573033707865), ('be', 0.006741573033707865), ('solvabl', 0.006741573033707865), ('solubl', 0.0056179775280898875), ('mean', 0.0056179775280898875), ('easili', 0.0056179775280898875), ('case', 0.0044943820224719105), ('situat', 0.0044943820224719105), ('understand', 0.0044943820224719105), ('problem', 0.0044943820224719105), ('puzzl', 0.0033707865168539327), ('exercis', 0.0033707865168539327), ('obviou', 0.0033707865168539327), ('insolv', 0.0033707865168539327), ('develop', 0.0033707865168539327), ('remain', 0.0033707865168539327), ('math', 0.0022471910112359553), ('task', 0.0022471910112359553), ('decid', 0.0022471910112359553), ('capabl', 0.0022471910112359553), ('exampl', 0.0011235955056179776), ('skill', 0.0011235955056179776), ('suscept', 0.0011235955056179776), ('mani', 0.0011235955056179776), ('unpleas', 0.0011235955056179776)]:
    #     sum += a[1]
    # print(sum)


def run(queue, okapi_func, label):
    okapi_mod_results = qrys.find_closest_docs(docs, okapi_func, doc_limit=doc_limit, query_limit=query_limit)
    okapi_mod_avg_map = score_fs.compute_avg_map(okapi_mod_results, relevant_docs)
    queue.append((label, okapi_mod_avg_map))

if __name__ == "__main__":
    # docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.all.1400", VectorType.DOCUMENTS, stemming_on=True)
    # qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.qry", VectorType.QUERIES, stemming_on=True)
    # relevant_docs = score_fs.read_human_judgement("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cranqrel", 1, 3)

    # docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/adi/ADI.ALL", VectorType.DOCUMENTS, stemming_on=True)
    # qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/adi/ADI.QRY", VectorType.QUERIES, stemming_on=True)
    # relevant_docs = score_fs.read_human_judgement("/Users/Eric/Desktop/Thesis/projects/datasets/adi/ADI.REL", 0, 0)  # DIFFERENT FORMAT

    docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/med/MED.ALL", VectorType.DOCUMENTS, stemming_on=True)
    qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/med/MED.QRY", VectorType.QUERIES, stemming_on=True)
    relevant_docs = score_fs.read_human_judgement_MED("/Users/Eric/Desktop/Thesis/projects/datasets/med/MED.REL", 1, 1)  # DIFFERENT FORMAT

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














