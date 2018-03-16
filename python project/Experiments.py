# Experiments and the location of main
# Eric LaBouve (elabouve@calpoly.edu)


from VectorCollection import VectorCollection, VectorType
from DistanceFunctions import CosineFunction, OkapiFunction, OkapiModFunction
from WordNet import WordNet
import ScoringFunctions as score_fs
import nltk, sys

if __name__ == "__main0__":
    qry = "papers on shock sound wave interaction"
    text = nltk.word_tokenize(qry)
    print(nltk.pos_tag(text))

    wn = WordNet()
    print(wn.get_sim_terms_rw('papers', depth=5))
    print(wn.get_sim_terms_rw('shock', depth=5))
    print(wn.get_sim_terms_rw('sound', depth=5))
    print(wn.get_sim_terms_rw('interaction', depth=5))


if __name__ == "__main__":
    docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.all.1400", VectorType.DOCUMENTS, stemming_on=True)
    # docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/test/documents2.txt", VectorType.DOCUMENTS)

    qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.qry",VectorType.QUERIES, stemming_on=True)
    # qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/test/queries2.txt", VectorType.QUERIES)

    relevant_docs = score_fs.read_human_judgement("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cranqrel", 1, 3)

    # Cosine Test
    docs.normalize(docs)
    qrys.normalize(docs)

    query_limit = 225
    doc_limit = 20

    cosine_results = qrys.find_closest_docs(docs, CosineFunction(docs), doc_limit=doc_limit, query_limit=query_limit)
    cosine_avg_map = score_fs.compute_avg_map(cosine_results, relevant_docs, query_limit=query_limit)
    print(cosine_avg_map)

    # okapi_func = OkapiFunction(docs)
    # okapi_results = qrys.find_closest_docs(docs, okapi_func, doc_limit=doc_limit, query_limit=query_limit)
    # okapi_avg_map = score_fs.compute_avg_map(okapi_results, relevant_docs)
    # print(okapi_avg_map)

    # okapi_func = OkapiModFunction(docs, is_adv_verb_pairs=True, is_adj_noun_pairs=True, is_early=True)
    # okapi_mod_results = qrys.find_closest_docs(docs, okapi_func, doc_limit=doc_limit, query_limit=query_limit)
    # okapi_mod_avg_map = score_fs.compute_avg_map(okapi_mod_results, relevant_docs)
    # print(okapi_mod_avg_map)

    # print('adv verb pairs')
    # influence = 1.1
    # best_influence = 0
    # best_map = 0
    # while influence <= 2.0:
    #     sys.stdout.write('Influence=' + str(influence) + ' ')
    #     sys.stdout.flush()
    #     okapi_func = OkapiModFunction(docs, is_adv_verb_pairs=True, adv_verb_pairs_influence=influence)
    #     okapi_mod_results = qrys.find_closest_docs(docs, okapi_func, doc_limit=doc_limit, query_limit=query_limit)
    #     okapi_mod_avg_map = score_fs.compute_avg_map(okapi_mod_results, relevant_docs)
    #     if okapi_mod_avg_map > best_map:
    #         best_map = okapi_mod_avg_map
    #         best_influence = influence
    #     print(okapi_mod_avg_map)
    #     influence += 0.1
    # print("Best influence = " + str(best_influence))
    # print("Best MAP = " + str(best_map))

