# Experiments and the location of main
# Eric LaBouve (elabouve@calpoly.edu)


from VectorCollection import VectorCollection, VectorType
from DistanceFunctions import CosineFunction
import json

if __name__ == "__main__":
    json_base_dir = '/Users/Eric/Desktop/Thesis/programs/java/json/'
    adj_list_json = json.load(open(json_base_dir + 'adjList.json'))
    edge_list_json = json.load(open(json_base_dir + 'edgeList.json'))
    id_to_label_json = json.load(open(json_base_dir + 'idToLabel.json'))
    label_to_id_json = json.load(open(json_base_dir + 'labelToId.json'))
    wf_vertex_db_json = json.load(open(json_base_dir + 'wfVertexDb.json'))

    docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.all.1400", False, False, 2, VectorType.DOCUMENTS)
    #docs = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/test/documents.txt", False, True, 2, VectorType.DOCUMENTS)

    qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/cran/cran.qry", False, False, 2, VectorType.QUERIES)
    #qrys = VectorCollection("/Users/Eric/Desktop/Thesis/projects/datasets/test/queries.txt", False, True, 2, VectorType.QUERIES)

    # Cosine Test
    docs.normalize(docs)
    qrys.normalize(docs)
    qv = qrys.id_to_textvector[1]
    cosine = CosineFunction(docs)
    cosine.set_query(qv)
    close_docs = qv.find_closest_docs(docs, cosine)[0:10]
    print(close_docs)
    # Compute Map

    #print(docs)
    #print(qrys)
