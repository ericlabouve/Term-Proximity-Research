import pysolr


if __name__ == "__main__":
    # solr_base = pysolr.Solr('http://localhost:8983/solr')
    solr = pysolr.Solr('http://localhost:8983/solr/myCore')

    # How you'd index data.
    solr.add([
        {
            "id": "3",
            "title": "A test document",
        },
    {
        "id": "doc_2",
        "title": "The Banana: Tasty or Dangerous?",
        "_doc": [
            { "id": "child_doc_1", "title": "peel" },
            { "id": "child_doc_2", "title": "seed" },
        ]
    }
    ])

    results = solr.search('document')

    # The ``Results`` object stores total results found, by default the top
    # ten most relevant results and any additional data like
    # facets/highlighting/spelling/etc.
    print("Saw "+str(len(results))+" result(s).")

    # Just loop over it to access the results.
    for result in results:
        print("The title is '{0}'.".format(result['title']))
