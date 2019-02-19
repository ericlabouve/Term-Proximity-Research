# Functions for scoring the distance functions
# Eric LaBouve (elabouve@calpoly.edu)

from collections import defaultdict
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import sys, json, os


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


# Calculate the weighted average recall for nDocs number of returned documents for all
# files in the specified directory, dir
# Weighted by the number of relevant documents in each query. (See finalPractice.pdf from CSC466 for equation)
# Equation:     Weighted Recall = Sum(Recall * (Num True Positives / Total Num Rel Docs Including Duplicate Docs))
def calc_all_recall(dir: str, nDir: list):
    # Open relevant list
    with open(dir + 'human_judgement.json') as f:
        human_judgement = json.load(f)

    # Loop through each file name in the directory
    for filename in os.listdir(dir):
        if filename == 'human_judgement.json' or '.txt' in filename:
            continue
        print(filename)
        with open(dir + filename) as f:
            results = json.load(f)

        # Get Total Num Rel Docs Including Duplicate Docs
        totalRel = 0
        for qryID in results.keys():
            # Only evaluate processed queries
            if qryID not in human_judgement:
                continue
            totalRel += len(human_judgement[qryID])

        # Calculate the recall at each level in nDir
        for level in nDir:
            weightedRecall = 0
            # For each query and the resulting returned documents
            for qryID, docList in results.items():
                # Only evaluate processed queries
                if qryID not in human_judgement:
                    continue
                # Obtain the |level| number of returned document
                levelDocs = docList[0:level]
                # Determine how many of these documents are relevant
                recall = len(intersection(levelDocs, human_judgement[qryID])) / len(human_judgement[qryID])
                weightedRecall += recall * (len(human_judgement[qryID]) / totalRel)
                # print('Recall at ' + str(level) + ' for ' + filename + ' on query ' + str(qryID) + ' = ' + str(recall))
            print('@' + str(level) + ' = ' + str(weightedRecall))
        print()



# ______________________ Graphing Functions ___________________


# A precision value at recall level i is calculated as follows:
#       p(ri) = from ri≤r≤10 max(p(r))
# num_queries - The number of queries to represent in this graph
def graph_precision_recall(num_queries, recall_buckets=20):
    style.use('fivethirtyeight')
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    with open('out\\train_cran\\lisa\\human_judgement.json') as f:
        human_judgement = json.load(f)
    with open('out\\train_cran\\lisa\\okapi_results.json') as f:
        okapi_results = json.load(f)
    with open('out\\train_cran\\lisa\\r3_ID14_results.json') as f:
        r3_ID14_results = json.load(f)

    # with open('out/cran/human_judgement.json') as f:
    #     human_judgement = json.load(f)
    # with open('out/cran/cosine_results.json') as f:
    #     cos_results = json.load(f)
    # with open('out/cran/okapi_results.json') as f:
    #     okapi_results = json.load(f)

    # with open('out/cran/okapi_isearly_results.json') as f:
    #     okapi_isearly_results = json.load(f)
    # with open('out/cran/okapi_isearlynoun_results.json') as f:
    #     okapi_isearlynoun_results = json.load(f)
    # with open('out/cran/okapi_isearlyverb_results.json') as f:
    #     okapi_isearlyverb_results = json.load(f)
    # with open('out/cran/okapi_isearlyadj_results.json') as f:
    #     okapi_isearlyadj_results = json.load(f)
    # with open('out/cran/okapi_isearlyadv_results.json') as f:
    #     okapi_isearlyadv_results = json.load(f)
    # with open('out/cran/okapi_isearlynounadj_results.json') as f:
    #     okapi_isearlynounadj_results = json.load(f)
    #
    # with open('out/cran/okapi_isearlynotnoun_results.json') as f:
    #     okapi_isearlynotnoun_results = json.load(f)
    # with open('out/cran/okapi_isearlynotverb_results.json') as f:
    #     okapi_isearlynotverb_results = json.load(f)
    # with open('out/cran/okapi_isearlynotadj_results.json') as f:
    #     okapi_isearlynotadj_results = json.load(f)
    # with open('out/cran/okapi_isearlynotadv_results.json') as f:
    #     okapi_isearlynotadv_results = json.load(f)
    #
    # with open('out/cran/okapi_isearlyq_results.json') as f:
    #     okapi_isearlyq_results = json.load(f)
    # with open('out/cran/okapi_isearlyqnoun_results.json') as f:
    #     okapi_isearlyqnoun_results = json.load(f)
    # with open('out/cran/okapi_isearlyqverb_results.json') as f:
    #     okapi_isearlyqverb_results = json.load(f)

    # cos_query_results, r1 = calc_pr_scores(cos_results, human_judgement, num_queries)
    okapi_query_results, r1 = calc_pr_scores(okapi_results, human_judgement, num_queries)
    r3_ID14_query_results, r2 = calc_pr_scores(r3_ID14_results, human_judgement, num_queries)

    # okapi_isearly_query_results, r2 = calc_pr_scores(okapi_isearly_results, human_judgement, num_queries)
    # okapi_isearlynoun_query_results, r2 = calc_pr_scores(okapi_isearlynoun_results, human_judgement, num_queries)
    # okapi_isearlyverb_query_results, r2 = calc_pr_scores(okapi_isearlyverb_results, human_judgement, num_queries)
    # okapi_isearlyadj_query_results, r2 = calc_pr_scores(okapi_isearlyadj_results, human_judgement, num_queries)
    # okapi_isearlyadv_query_results, r2 = calc_pr_scores(okapi_isearlyadv_results, human_judgement, num_queries)
    # okapi_isearlynounadj_query_results, r2 = calc_pr_scores(okapi_isearlynounadj_results, human_judgement, num_queries)
    #
    # okapi_isearlynotnoun_query_results, r2 = calc_pr_scores(okapi_isearlynotnoun_results, human_judgement, num_queries)
    # okapi_isearlynotverb_query_results, r2 = calc_pr_scores(okapi_isearlynotverb_results, human_judgement, num_queries)
    # okapi_isearlynotadj_query_results, r2 = calc_pr_scores(okapi_isearlynotadj_results, human_judgement, num_queries)
    # okapi_isearlynotadv_query_results, r2 = calc_pr_scores(okapi_isearlynotadv_results, human_judgement, num_queries)
    #
    # okapi_isearlyq_query_results, r2 = calc_pr_scores(okapi_isearlyq_results, human_judgement, num_queries)
    # okapi_isearlyqnoun_query_results, r2 = calc_pr_scores(okapi_isearlyqnoun_results, human_judgement, num_queries)
    # okapi_isearlyqverb_query_results, r2 = calc_pr_scores(okapi_isearlyqverb_results, human_judgement, num_queries)

    max_recall = max([r1, r2])
    bucket_s = max_recall / recall_buckets  # Length between each tick on the x axis
    xs = [float('%.2f' % (bucket_s * x)) for x in range(recall_buckets + 1)]
    print(max_recall)

    # Algorithm and its label
    data_lists = [
                  # (okapi_isearly_query_results, 'is early'),
                  # (okapi_isearlynoun_query_results, 'is early noun'),
                  # (okapi_isearlyverb_query_results, 'is early verb'),
                  # (okapi_isearlyadj_query_results, 'is early adj'),
                  # (okapi_isearlyadv_query_results, 'is early adv'),
#                  (okapi_isearlynounadj_query_results, 'is early noun+adj'),

#                  (okapi_isearlynotnoun_query_results, 'is early nnoun'),
#                  (okapi_isearlynotverb_query_results, 'is early nverb'),
#                  (okapi_isearlynotadj_query_results, 'is early nadj'),
#                  (okapi_isearlynotadv_query_results, 'is early nadv'),

#                  (okapi_isearlyq_query_results, 'is early query'),
#                  (okapi_isearlyqnoun_query_results, 'is early query noun'),
#                  (okapi_isearlyqverb_query_results, 'is early query verb'),

#                  (cos_query_results, 'cosine'),
                  (okapi_query_results, 'Okapi BM25'),
                  (r3_ID14_query_results, 'Modification')
                  ]

    m = ['o', '^']
    i = 0
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
        ax1.plot(xs, ys, marker=m[i], markersize=6.5, label=label, linewidth=1.5)
        i += 1

    font = {'family': 'normal',
            'size': 16}
    plt.rc('font', **font)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.18, .7])
    plt.xlim([0.0, .95])#max_recall + .05])
    # plt.title('PR Curves on Test Benchmark')
    plt.subplots_adjust(bottom=0.09)
    plt.legend()
    # plt.xticks(xs[:len(xs) - 2])
    # plt.xticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9])
    plt.xticks([0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9])
    ys = [.2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7]
    # ys = [.2, .3, .4, .5, .6, .7]
    plt.yticks(ys)
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

# Returns a mapping from {Query Ids : [Relevant Doc Ids]}
# path - File path
# best Beginning range for human judgement
# worst Ending range for human judgement
def read_human_judgement_MED(file_path, best_score, worst_score):
    relevant_docs = defaultdict(list)
    with open(file_path) as file:
        for line in file:
            items = line.split()
            query_num = int(items[0])
            doc_num = int(items[2])
            degree_rel = int(items[3])
            # Check if degree of relevance is in our bounds
            if best_score <= degree_rel <= worst_score:
                relevant_docs[query_num].append(doc_num)
    return relevant_docs


def read_human_judgement_TIME(file_path):
    relevant_docs = defaultdict(list)
    with open(file_path) as file:
        for line in file:
            items = line.split()
            query_num = int(items[0])
            for doc_num in items[1:]:
                relevant_docs[query_num].append(int(doc_num))
    return relevant_docs













