# Goal is to count the number of possible query expansion terms discovered for each benchmark

from WordNet import WordNet
import re
from pathlib import Path
from nltk.stem import PorterStemmer


stemmer = PorterStemmer()
wn = WordNet()


for path in [Path('../datasets/cran/cran.qry'), Path('../datasets/adi/ADI.QRY'),
             Path('../datasets/med/MED.QRY'), Path('../datasets/time/TIME_clean.QUE'), Path('../datasets/lisa/LISA.QUE')]:
    qTermsList = []  # A list of sets where each set contains the terms for a single query
    with open(path, 'r') as qfile:
        oneQuerysTerms = set()  # A set containing a single query's terms
        for line in qfile:
            if '.I' in line:
                qTermsList.append(oneQuerysTerms)
                oneQuerysTerms = set()
            elif '.I' not in line or '.W' not in line:
                oneQuerysTerms |= set(re.split('[^a-zA-Z]+', line))
        qTermsList.append(oneQuerysTerms)
    qTermsList_s = []
    for oneQuerysTerms in qTermsList:
        oneQuerysTermsSet = set()
        for term in oneQuerysTerms:
            stemWord = stemmer.stem(term)
            if len(stemWord) > 0:
                oneQuerysTermsSet |= set([stemWord])
        qTermsList_s.append(oneQuerysTermsSet)

    # Get all the possible substitution terms
    qCount = 0
    table = {}
    for q in qTermsList_s:
        keys = table.keys()
        for t in q:
            if t not in keys:
                subs = wn.get_sim_terms_rw(t)
                subs_stem = wn.stem(stemmer, t, subs)
                table[t] = subs_stem
        qCount += 1
        if qCount == 2:
            break


    # Calculate how many terms are found for expansion (not unique)
    # Calculate how many expansion terms are available
    keys = table.keys()  # The set of keys from the table
    termsThatCanBeExpanded = 0
    expTerms = 0
    for q in qTermsList_s:
        for t in q:
            if t in keys:
                termsThatCanBeExpanded += 1
                posExpTerms = table[t]
                for posExpTerm in posExpTerms:
                    # Compare the similarity score to the minimum sim score during experiments
                    if posExpTerm[1] > 0.02:
                        expTerms += 1

    print(str(path) + " has " + str(termsThatCanBeExpanded) + " terms that can be expanded and " + str(
        expTerms) + " total expansion terms. Ratio = " + str(expTerms / termsThatCanBeExpanded))