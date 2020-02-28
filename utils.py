import json

import nltk


def dump_jsonl_file(objs, path):
    with open(path, 'w') as fh:
        for r in objs:
            fh.write(json.dumps(r) + "\n")


def dumps_jsonl(objs):
    for r in objs:
        print(json.dumps(r))


def load_jsonl_file(path):
    with open(path, 'r') as fh:
        return [json.loads(line) for line in fh]


def load_parse_trees(doc_id, parse_strings):
    results = []
    for sent_id, sent in enumerate(parse_strings):
        try:
            ptree = nltk.Tree.fromstring(sent.strip())
            if not ptree.leaves():
                print('Failed on empty tree')
                results.append(None)
            else:
                results.append(ptree)
        except ValueError:
            print('Failed to parse doc {} idx {}'.format(doc_id, sent_id))
            results.append(None)
    return doc_id, results
