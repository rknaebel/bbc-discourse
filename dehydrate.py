import plac

from utils import dumps_jsonl, load_jsonl_file


def remove_textual_data(relation):
    return {
        'Arg1': relation['Arg1']['TokenList'],
        'Arg2': relation['Arg2']['TokenList'],
        'Connective': relation['Connective']['TokenList'],
        'DocID': relation['DocID']
    }


def main(relation_path):
    bbc_relations = load_jsonl_file(relation_path)
    dumps_jsonl(map(remove_textual_data, bbc_relations))


if __name__ == '__main__':
    plac.call(main)
