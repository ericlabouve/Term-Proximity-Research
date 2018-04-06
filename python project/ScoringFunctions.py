# Functions for scoring the distance functions
# Eric LaBouve (elabouve@calpoly.edu)

from collections import defaultdict
from matplotlib import style
import matplotlib.pyplot as plt
import sys, json


# ______________________ Precision, Recall, MAP, and F-Score Functions ___________________

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


def precision(num_rel: int, total_ret: int):
    return num_rel / total_ret


def f1_score(prec, recal):
    return (2 * prec * recal) / (prec + recal)


# Returns the intersection of l1 and l2 as a set
# l1 - The first list
# l2 - The second list
def intersection(l1: list, l2: list) -> set:
    keys_v1 = set(l1)
    keys_v2 = set(l2)
    return keys_v1 & keys_v2  # items that v1 and v2 share


# ______________________ Graphing Functions ___________________


# A precision value at recall level i is calculated as follows:
#       p(ri) = from ri≤r≤10 max(p(r))
# num_queries - The number of queries to represent in this graph
def graph_precision_recall(num_queries, recall_buckets=10):
    style.use('fivethirtyeight')
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    with open('out/human_judgement.json') as f:
        human_judgement = json.load(f)
    with open('out/cosine_results.json') as f:
        cos_results = json.load(f)
    with open('out/okapi_results.json') as f:
        okapi_results = json.load(f)
    with open('out/okapi_isearly_results.json') as f:
        okapi_isearly_results = json.load(f)

    cos_query_results, r1 = calc_pr_scores(cos_results, human_judgement, num_queries)
    okapi_query_results, r2 = calc_pr_scores(okapi_results, human_judgement, num_queries)
    okapi_isearly_query_results, r2 = calc_pr_scores(okapi_isearly_results, human_judgement, num_queries)

    max_recall = max([r1, r2])
    bucket_s = max_recall / recall_buckets  # Length between each tick on the x axis
    xs = [float('%.2f' % (bucket_s * x)) for x in range(recall_buckets + 1)]
    print(max_recall)
    # Algorithm and its label
    data_lists = [(cos_query_results, 'cosine'), (okapi_query_results, 'okapi'), (okapi_isearly_query_results, 'is early')]
    # Loop through each IR algorithm
    for score_list, label in data_lists:
        ys = [0 for x in range(recall_buckets + 1)]
        for query in score_list:
            # Fill out all precision values
            for idx in range(recall_buckets):
                # Find the max precision in score_list with recall >= cur_thresh
                p = max(query[idx:])
                ys[idx] += p[0]
        ys = [y / len(score_list) for y in ys]
        ax1.plot(xs, ys, label=label, linewidth=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, .55])
    plt.xlim([0.0, max_recall + .05])
    plt.title('Precision-Recall Curves')
    plt.subplots_adjust(bottom=0.09)
    plt.legend()
    plt.xticks(xs)
    plt.show()


# human_judgement - map {Query Ids : [Relevant Doc Ids]}
# alg_results -     map {Query id : [Doc Ids]} where Doc Ids are sorted in order of relevance
def calc_pr_scores(alg_results: list, human_judgement: list, num_queries) -> (list, int):
    query_results = []  # Precision Recall results for each query: list(list(2-tuple))
    counter = 0
    max_recall = 0
    # Loop through each cosine result and calculate precision and recall scores
    for qry_id, results_list in alg_results.items():
        counter += 1
        if qry_id not in human_judgement:
            continue
        relevant_list = human_judgement[qry_id]
        pr_results = []  # Precision Recall results for this query
        # Calculate Precision and Recall
        for i in range(len(results_list)):
            predict_docs = results_list[0:i + 1]
            num_correct_docs = len(intersection(predict_docs, relevant_list))

            # Calculate precision
            p = num_correct_docs / (i + 1)
            # Calculate recall
            r = num_correct_docs / len(relevant_list)

            pr_results.append((p, r))

            if r > max_recall:
                max_recall = r
        query_results.append(pr_results)

        if counter >= num_queries:
            break
    return query_results, max_recall


# ______________________ I/O Functions ___________________

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

















