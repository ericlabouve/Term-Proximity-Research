
from VectorCollection import VectorCollection, VectorType
import json

if __name__ == "__main__":
    json_base_dir = '/Users/Eric/Desktop/Thesis/programs/java/json/'
    adj_list_json = json.load(open(json_base_dir + 'adjList.json'))
    edge_list_json = json.load(open(json_base_dir + 'edgeList.json'))
    id_to_label_json = json.load(open(json_base_dir + 'idToLabel.json'))
    label_to_id_json = json.load(open(json_base_dir + 'labelToId.json'))
    wf_vertex_db_json = json.load(open(json_base_dir + 'wfVertexDb.json'))

    #docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.all.1400", False, False, 2, VectorType.DOCUMENTS)
    #docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/test/documents.txt", False, True, 2, VectorType.DOCUMENTS)

    qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.qry", False, True, 2, VectorType.DOCUMENTS)
    #qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/test/queries.txt", False, True, 2, VectorType.DOCUMENTS)

    #print(docs)
    print(qrys)
