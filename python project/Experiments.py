# Experiments and the location of main
# Eric LaBouve (elabouve@calpoly.edu)


from VectorCollection import VectorCollection, VectorType
from DistanceFunctions import CosineFunction, OkapiFunction, OkapiModFunction
import ScoringFunctions as score_fs
import json

if __name__ == "__main__":

    json_base_dir = '/Users/Eric/Desktop/Thesis/programs/java/json/'
    adj_list_json = json.load(open(json_base_dir + 'adjList.json'))
    edge_list_json = json.load(open(json_base_dir + 'edgeList.json'))
    id_to_label_json = json.load(open(json_base_dir + 'idToLabel.json'))
    label_to_id_json = json.load(open(json_base_dir + 'labelToId.json'))
    wf_vertex_db_json = json.load(open(json_base_dir + 'wfVertexDb.json'))

    docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.all.1400", VectorType.DOCUMENTS)
    #docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/test/documents2.txt", VectorType.DOCUMENTS)

    qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.qry",VectorType.QUERIES)
    #qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/test/queries2.txt", VectorType.QUERIES)

    relevant_docs = score_fs.read_human_judgement("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cranqrel", 1, 3)

    # Cosine Test
    docs.normalize(docs)
    qrys.normalize(docs)

    query_limit = 225

#    cosine_results = qrys.find_closest_docs(docs, CosineFunction(docs), doc_limit=20)
#    cosine_avg_map = score_fs.compute_avg_map(cosine_results, relevant_docs, query_limit=225)
#    print(cosine_avg_map)

    okapi_results = qrys.find_closest_docs(docs, OkapiFunction(docs), doc_limit=20, query_limit=query_limit)
    okapi_avg_map = score_fs.compute_avg_map(okapi_results, relevant_docs)
    print(okapi_avg_map)

    okapi_mod_results = qrys.find_closest_docs(docs, OkapiModFunction(docs, is_early=True), doc_limit=20, query_limit=query_limit)
    okapi_mod_avg_map = score_fs.compute_avg_map(okapi_mod_results, relevant_docs)
    print(okapi_mod_avg_map)


