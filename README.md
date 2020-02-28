# BBC Discourse Relations

In this repository, we provide files that contain information about the processed BBC News Corpus and its extracted relations.
The uploaded data is made anonymous.
Therefore, we also provide scripts for producing those files as well as reconstructing the original extracted relations.
If you make use of these datasets please consider citing the publication:
 
R. Knaebel and M. Stede. "Semi-Supervised Tri-Training for Explicit Discourse Argument Expansion", Proc. LREC 2020
[[PDF]](LINK) [[BibTeX]](LINK)

## Create BBC Corpus
For corpus preparation, we refer to the `make_corpus.py` script.
It gets the path to one of the downloaded raw [BBC corpora](http://mlg.ucd.ie/datasets/bbc.html) and writes all information into one json file.
The format is comparable to the CoNLL2016 format of the shared task.
Corpus links:
- bbc: http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip
- bbcsport: http://mlg.ucd.ie/files/datasets/bbcsport-fulltext.zip
```
python3 make_corpus.py CORPUS_PATH JSON_PATH.json
```

## Dehydrate
For removing textual information, we use the `dehydrate.py` script.
It returns a flattened json structure that contains only *TokenList* information and the corresponding document id.
```
python3 dehydrate.py RELATIONS_PATH > RELATION_ID.json
```

## Hydrate
For back conversion, we use the `hydrate.py` script.
It combines the extracted TokenLists with the corpus file and thus reconstructs the original extraction.
```
python3 hydrate.py JSON_PATH.json RELATION_ID.json > RELATION_FULL.json
```
