import json
import multiprocessing
import os
import re
from glob import glob

import benepar
import nltk
import plac
import spacy
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import dump_jsonl_file

regex_ws = re.compile(r'\s+')


def load_corpus(path, file_name):
    nlp = spacy.load('en')
    nlp.tokenizer = nlp.tokenizer.tokens_from_list
    parser = benepar.Parser('benepar_en2')
    tbwt = nltk.TreebankWordTokenizer()
    tbwd = nltk.treebank.TreebankWordDetokenizer()
    documents = {}
    with open("{}/{}.json".format(path, file_name), 'r') as corpus_fh:
        for raw_doc in tqdm(corpus_fh.readlines()):
            try:
                doc = convert_to_conll(json.loads(raw_doc), nlp, parser, tbwt, tbwd)
                documents[doc['DocID'].strip()] = doc
            except Exception as e:
                print('Failed to parse document:')
                print(e)
                continue
    return documents


def convert_to_conll(document, nlp, parser, tbwt, tbwd):
    text = " ".join([
        tbwd.detokenize(tbwt.tokenize(s)).replace(r'\"', '``').replace('"', "''")
        for s in nltk.sent_tokenize(document['Text'])])
    sentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)]
    ptrees = parser.parse_sents(sentences)
    doc = nlp.pipe(sentences)
    res = []
    offset = 0
    for sent, ptree in tqdm(zip(doc, ptrees), total=len(sentences), desc='sentences', leave=False):
        words = []
        for tok in sent:
            token_offset = text.find(tok.text, offset)
            offset = token_offset + len(tok.text)
            words.append((tok.text, {
                'CharacterOffsetBegin': token_offset,
                'CharacterOffsetEnd': offset,
                'Linkers': [],
                'PartOfSpeech': tok.tag_,
                'SimplePartOfSpeech': tok.pos_,
                'Lemma': tok.lemma_,
                'Shape': tok.shape_,
                'EntIOB': tok.ent_iob_,
                'EntType': tok.ent_type_
            }))

        sentence_conll = {
            'dependencies': [(t.dep_, "{}-{}".format(t.head.text, t.head.i + 1), "{}-{}".format(t.text, t.i + 1)) for t
                             in
                             sent],
            'parsetree': ptree._pformat_flat('', '()', False),
            'words': words,
        }
        res.append(sentence_conll)
    return {
        'Corpus': document['Corpus'],
        'DocID': document['DocID'],
        'raw': document['Text'],
        'text': text,
        'sentences': res,
    }


def extract(path):
    with open(path, 'r', encoding='latin-1') as fh:
        text = fh.read()
    title = text.split('\n\n')[0]
    corpus, topic, id = path.split('/')[-3:]
    text = regex_ws.sub(r' ', text.strip()[len(title) + 2:])
    return {
        'Meta': {
            'title': title,
            'topic': topic,
        },
        'Text': text,
        'Corpus': corpus,
        'DocID': corpus + '-{}-{}'.format(topic, id[:-len('.txt')])
    }


def get_data(path):
    news_files = glob(os.path.join(path, 'bbc/*/*.txt'))

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        news_json = pool.map(extract, news_files, chunksize=8)

    return news_json


def get_sports_data(path):
    news_files = glob(os.path.join(path, 'bbcsport/*/*.txt'))

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        news_json = pool.map(extract, news_files, chunksize=8)

    return news_json


def load_unlabeled_corpora(data_path, outdir):
    print('load BBC corpus')
    bbc = get_data(data_path)
    print('load BBC Sports corpus')
    bbc_sports = get_sports_data(data_path)
    dump_jsonl_file(bbc, os.path.join(outdir, 'bbc_corpus_featured.json'))
    dump_jsonl_file(bbc_sports, os.path.join(outdir, 'bbcsport_corpus_featured.json'))


def remove_textual_data(relation):
    return {
        'Arg1': relation['Arg1']['TokenList'],
        'Arg2': relation['Arg2']['TokenList'],
        'Connective': relation['Connective']['TokenList'],
        'DocID': relation['DocID']
    }


def main(data_path, parses_path):
    load_unlabeled_corpora(data_path, parses_path)

    docs = load_corpus(parses_path, 'bbc_corpus_featured')
    print('save bcc corpus')
    json.dump(docs, open("{}/{}_new.json".format(parses_path, 'bbc_corpus_featured'), 'w'))

    load_corpus(parses_path, 'bbcsport_corpus_featured')
    print('save bccsport corpus')
    json.dump(docs, open("{}/{}_new.json".format(parses_path, 'bbcsport_corpus_featured'), 'w'))


if __name__ == '__main__':
    plac.call(main)
