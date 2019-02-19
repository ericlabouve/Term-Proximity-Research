# Get the average document length
inBody = False
numDocs = 0
totalNumTerms = 0
with open('lisa_clean.all') as f:
    for line in f:
        if '.I' in line:
            inBody = False
            numDocs += 1
        elif '.W' in line:
            inBody = True
        else:
            totalNumTerms += len(line.split())
print(totalNumTerms/numDocs)

# Get the average query length
inBody = False
numQrys = 0
totalNumQTerms = 0
stop_words = {"a", "about", "above", "all", "along","also", "although", "am", "an", "and", "any", "are", "aren't", "as", "at","be", "because", "been", "but", "by", "can", "cannot", "could", "couldn't","did", "didn't", "do", "does", "doesn't", "e.g.", "either", "etc", "etc.","even", "ever", "enough", "for", "from", "further", "get", "gets", "got", "had", "have","hardly", "has", "hasn't", "having", "he", "hence", "her", "here","hereby", "herein", "hereof", "hereon", "hereto", "herewith", "him","his", "how", "however", "i", "i.e.", "if", "in", "into", "it", "it's", "its","me", "more", "most", "mr", "my", "near", "nor", "now", "no", "not", "or", "on", "of", "onto","other", "our", "out", "over", "really", "said", "same", "she","should", "shouldn't", "since", "so", "some", "such","than", "that", "the", "their", "them", "then", "there", "thereby","therefore", "therefrom", "therein", "thereof", "thereon", "thereto","therewith", "these", "they", "this", "those", "through", "thus", "to","too", "under", "until", "unto", "upon", "us", "very", "was", "wasn't","we", "were", "what", "when", "where", "whereby", "wherein", "whether","which", "while", "who", "whom", "whose", "why", "with", "without","would", "you", "your", "yours", "yes"}
with open('LISA.QUE') as f:
    for line in f:
        if '.I' in line:
            inBody = False
            numQrys += 1
        elif '.W' in line:
            inBody = True
        else:
            for qTerm in line.split():
                if qTerm not in stop_words:
                    totalNumQTerms += 1
print(totalNumQTerms/numQrys)

# Get the average number of related documents
numQueries = 0
numRelDocs = 0
with open('LISARJ.NUM') as f:
    for line in f:
        numQueries += 1
        numRelDocs += len(line.split()) - 1
print(numRelDocs/numQueries)