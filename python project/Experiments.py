# Experiments and the location of main
# Eric LaBouve (elabouve@calpoly.edu)


from VectorCollection import VectorCollection, VectorType
from DistanceFunctions import CosineFunction
import ScoringFunctions as score_fs
import json

if __name__ == "__main__":

    json_base_dir = '/Users/Eric/Desktop/Thesis/programs/java/json/'
    adj_list_json = json.load(open(json_base_dir + 'adjList.json'))
    edge_list_json = json.load(open(json_base_dir + 'edgeList.json'))
    id_to_label_json = json.load(open(json_base_dir + 'idToLabel.json'))
    label_to_id_json = json.load(open(json_base_dir + 'labelToId.json'))
    wf_vertex_db_json = json.load(open(json_base_dir + 'wfVertexDb.json'))

    docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.all.1400", False, False, 2, VectorType.DOCUMENTS)
    #docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/test/documents2.txt", False, False, 2, VectorType.DOCUMENTS)

    qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.qry", False, False, 2, VectorType.QUERIES)
    #qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/test/queries2.txt", False, False, 2, VectorType.QUERIES)

    relevant_docs = score_fs.read_human_judgement("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cranqrel", 1, 3)

    # Cosine Test
    docs.normalize(docs)
    qrys.normalize(docs)

    results = qrys.find_closest_docs(docs, CosineFunction(docs), doc_limit=20)
    avg_map = score_fs.compute_avg_map(results, relevant_docs, query_limit=225)
    print(avg_map)

    java_results = [13, 184, 12, 51, 486, 1268, 327, 435, 746, 875, 665, 686, 359, 878, 494, 14, 1144, 1186, 332, 154]
    print(score_fs.compute_map(java_results, relevant_docs[1]))
    print(score_fs.compute_map(results[1], relevant_docs[1]))

