import click
import nmslib
import os
import pandas as pd
import random
import sys
import tensorflow as tf
import tensorflow_hub as hub

from ranking_metrics import average_precision
from ranking_metrics import ndcg_at_k

file_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(file_dir, '..'))
sys.path.append(base_dir)

from officeanswers.ingestion import DSSMPrepare
from matchzoo import datapack
from matchzoo import generators
from sklearn.metrics.pairwise import cosine_similarity


class ExpandedPath(click.Path):
    def convert(self, value, *args, **kwargs):
        value = os.path.expanduser(value)
        return super(ExpandedPath, self).convert(value, *args, **kwargs)


@click.command()
@click.argument('corpus_path',
                type=ExpandedPath(exists=True))
def word_embed_metrics(corpus_path: str) -> None:
    prep = DSSMPrepare()
    raw_ques, raw_docs, rels = prep.from_one_corpus(corpus_path)

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    embed = hub.Module(module_url)
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    with tf.Session(config=config) as session:
        session.run([tf.global_variables_initializer(),
                     tf.tables_initializer()])

        docs = list(raw_docs.values())
        docs_embeds = session.run(embed(docs))
        right = dict([(did, d) for did, d in zip(raw_docs.keys(),
                                                 docs_embeds)])

        ques = list(raw_ques.values())
        ques_embeds = session.run(embed(ques))
        left = dict([(qid, q) for qid, q in zip(raw_ques.keys(),
                                                ques_embeds)])

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
        scores = []
        for d in c:
            scores.append(
                cosine_similarity(
                    left[qid].reshape(1, -1),
                    d.reshape(1, -1)))
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
    word_embed_metrics()
