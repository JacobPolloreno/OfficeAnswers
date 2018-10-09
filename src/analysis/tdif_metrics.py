import click
import pandas as pd
import os
import random
import sys

from ranking_metrics import average_precision
from ranking_metrics import ndcg_at_k

file_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(file_dir, '..'))
sys.path.append(base_dir)


from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import Similarity
from officeanswers.ingestion import DSSMPrepare
from nltk.tokenize import word_tokenize


class ExpandedPath(click.Path):
    def convert(self, value, *args, **kwargs):
        value = os.path.expanduser(value)
        return super(ExpandedPath, self).convert(value, *args, **kwargs)


@click.command()
@click.argument('corpus_path',
                type=ExpandedPath(exists=True))
def tdif_metrics(corpus_path: str) -> None:
    prep = DSSMPrepare()
    raw_ques, raw_docs, rels = prep.from_one_corpus(corpus_path)

    docs = [[w.lower() for w in word_tokenize(text)]
            for text in raw_docs.values()]
    ques = [[w.lower() for w in word_tokenize(text)]
            for text in raw_ques.values()]
    docs = docs + ques
    dictionary = Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    tf_idf = TfidfModel(corpus)

    right = {}
    for did, doc_text in raw_docs.items():
        dense_input = [w.lower() for w in word_tokenize(doc_text)]
        dense_input = dictionary.doc2bow(dense_input)
        dense_input = tf_idf[dense_input]
        right[did] = dense_input

    left = {}
    for qid, ques_text in raw_ques.items():
        dense_input = [w.lower() for w in word_tokenize(doc_text)]
        dense_input = dictionary.doc2bow(dense_input)
        dense_input = tf_idf[dense_input]
        left[qid] = dense_input

    relations = pd.DataFrame(rels,
                             columns=['label', 'id_left', 'id_right'])
    res = {}
    res['MAP'] = 0.0
    res['NDCG@3'] = 0.0
    res['NDCG@5'] = 0.0
    num_valid = 0
    for group in relations.groupby('id_left'):
        qid, data = group
        dids = data['id_right'].values.tolist()
        labels = data['label'].values.tolist()
        c = [right[did] for did in dids]
        sims = Similarity('tf_idf', tf_idf[c],
                          num_features=len(dictionary))
        scores = sims[left[qid]]
        rank = list(zip(labels, scores))
        random.shuffle(rank)
        rank = sorted(rank, key=lambda x: x[1], reverse=True)
        rank = [float(r[0]) for r in rank]
        res['MAP'] += average_precision(rank)
        res['NDCG@3'] += ndcg_at_k(rank, 3)
        res['NDCG@5'] += ndcg_at_k(rank, 5)
        num_valid += 1

    click.echo('\t'.join(
        [f"{k}={v / num_valid:.3f}" for k, v in res.items()]))


if __name__ == "__main__":
    tdif_metrics()
