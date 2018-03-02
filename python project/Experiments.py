# Experiments and the location of main
# Eric LaBouve (elabouve@calpoly.edu)


from VectorCollection import VectorCollection, VectorType
from DistanceFunctions import CosineFunction, OkapiFunction, OkapiModFunction
from WordNet import WordNet
import ScoringFunctions as score_fs
import nltk


if __name__ == "__main__":
    qry = "what similarity laws must be obeyed when constructing aeroelastic models of heated high speed car."
    text = nltk.word_tokenize(qry)
    tagged = nltk.pos_tag(text)

    wn = WordNet()
    synonyms = wn.get_syns(tagged[14])


if __name__ == "__main2__":
    docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.all.1400", VectorType.DOCUMENTS)
    #docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/test/documents2.txt", VectorType.DOCUMENTS)

    qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.qry",VectorType.QUERIES)
    #qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/test/queries2.txt", VectorType.QUERIES)

    relevant_docs = score_fs.read_human_judgement("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cranqrel", 1, 3)

    # Cosine Test
    docs.normalize(docs)
    qrys.normalize(docs)

    query_limit = 20

    cosine_results = qrys.find_closest_docs(docs, CosineFunction(docs), doc_limit=20)
    cosine_avg_map = score_fs.compute_avg_map(cosine_results, relevant_docs, query_limit=225)
    print(cosine_avg_map)

    okapi_results = qrys.find_closest_docs(docs, OkapiFunction(docs), doc_limit=20, query_limit=query_limit)
    okapi_avg_map = score_fs.compute_avg_map(okapi_results, relevant_docs)
    print(okapi_avg_map)

    okapi_mod_results = qrys.find_closest_docs(docs, OkapiModFunction(docs, is_early=True), doc_limit=20, query_limit=query_limit)
    okapi_mod_avg_map = score_fs.compute_avg_map(okapi_mod_results, relevant_docs)
    print(okapi_mod_avg_map)


