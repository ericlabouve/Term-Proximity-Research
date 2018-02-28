# Functions for scoring the distance functions
# Eric LaBouve (elabouve@calpoly.edu)

from collections import defaultdict
import sys


# Computes the average for ALL MAP scores given in the results map parameter
# results - A map containing relevant documents for each query computed using find_closest_docs
# relevant - A map containing human judgement computed from read_human_judgement
# query_limit - An upper limit for the number of queries to process
def compute_avg_map(results: map, relevant: map, query_limit=sys.maxsize) -> float:
    map_sum = 0
    num_computed = 0
    for qry_id, results_list in results.items():
        # If we have not reached a user defined limit and there exist relevant ids for this query
        if num_computed < query_limit:
            if len(relevant[qry_id]) > 0:
                relevant_list = relevant[qry_id]
                map_sum += compute_map(results_list, relevant_list)
                num_computed += 1
        else:
            break
    return map_sum / num_computed


# Mean Average Precision is computed by taking the precision in order from
# start to end of the doc_id_results list and comparing that to the relevant_ids.
# doc_id_results - List of document ids outputted by the distance function.
# relevant_ids - List of related document ids for the query.
def compute_map(doc_id_results: list, relevant_ids: list) -> float:
    precision_sum = 0
    for index, doc_id in enumerate(doc_id_results):
        if doc_id in relevant_ids:
            intersect = len(intersection(doc_id_results[0:index + 1], relevant_ids))
            precision_sum += precision(intersect, index + 1)
    return precision_sum / len(relevant_ids)


def precision(num_rel, total_ret):
    return num_rel / total_ret


# Returns the intersection of l1 and l2 as a set
# l1 - The first list
# l2 - The second list
def intersection(l1: list, l2: list) -> set:
    keys_v1 = set(l1)
    keys_v2 = set(l2)
    return keys_v1 & keys_v2  # items that v1 and v2 share


# Returns a mapping from {Query Ids : [Relevant Doc Ids]}
# path - File path
# best Beginning range for human judgement
# worst Ending range for human judgement
def read_human_judgement(file_path, best_score, worst_score):
    relevant_docs = defaultdict(list)
    with open(file_path) as file:
        for line in file:
            items = line.split()
            query_num = int(items[0])
            doc_num = int(items[1])
            degree_rel = int(items[2])
            # Check if degree of relevance is in our bounds
            if best_score <= degree_rel <= worst_score:
                relevant_docs[query_num].append(doc_num)
    return relevant_docs

















