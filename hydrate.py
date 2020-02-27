import re

import plac
import ujson as json

from utils import load_jsonl_file, dumps_jsonl

regex_ws = re.compile(r'\s+')


def load_corpus(path):
    documents = json.load(open(path, 'r'))
    return documents


def hydrate(parses, relation):
    doc = parses.get(relation['DocID'])
    text = doc.get('text', '') if doc else ''
    return {
        'Arg1': {
            'TokenList': relation['Arg1'],
            'RawText': ' '.join([text[t[0]:t[1]] for t in relation['Arg1']]),
        },
        'Arg2': {
            'TokenList': relation['Arg2'],
            'RawText': ' '.join([text[t[0]:t[1]] for t in relation['Arg2']]),
        },
        'Connective': {
            'TokenList': relation['Connective'],
            'RawText': ' '.join([text[t[0]:t[1]] for t in relation['Connective']]),
        },
        'DocId': relation['DocID']
    }


def main(parses_path, relation_path):
    corpus = load_corpus(parses_path)
    relations = load_jsonl_file(relation_path)
    dumps_jsonl(map(lambda r: hydrate(corpus, r), relations))


if __name__ == '__main__':
    plac.call(main)
