"""Office Answers CLI

"""

import click
import logging
import numpy as np
import os

from officeanswers.preprocess import build_document_embeddings
from officeanswers.preprocess import prepare_and_preprocess
from officeanswers.model import get_inference_model
from officeanswers.model import train as model_train
from officeanswers.model import predict as model_predict
from officeanswers.model import evaluate as model_evaluate
from officeanswers.util import Config

from matchzoo import engine

logger = logging.getLogger(__name__)


class ExpandedPath(click.Path):
    def convert(self, value, *args, **kwargs):
        value = os.path.expanduser(value)
        return super(ExpandedPath, self).convert(value, *args, **kwargs)


@click.group()
@click.argument('model_config_file',
                type=ExpandedPath(exists=True))
@click.option('--debug', is_flag=True)
@click.pass_context
def cli(ctx, debug, model_config_file):
    """CLI for the office assistant model.

    Example Usage:\n
        python src/main.py config/model.config train
    """
    ctx.obj['DEBUG'] = debug
    ctx.obj['CONFIG'] = Config()
    ctx.obj['CONFIG'].from_json_file(model_config_file)

    logger.info("Loaded configuration file...")


@cli.command()
@click.pass_context
@click.option('--save', '-s', is_flag=True, help="Save corpus and relations")
def prepare(ctx, save: bool=False):
    prepare_and_preprocess(ctx.obj['CONFIG'], save)


@cli.command()
@click.pass_context
def train(ctx):
    model_train(ctx.obj['CONFIG'])


@cli.command()
@click.pass_context
@click.option('--num_largest', '-n', type=int, default=5)
def predict(ctx, num_largest):
    logger.info('Loading model...')

    config = ctx.obj['CONFIG']
    pr_dir = config.paths['processed_dir']
    model_type = config.model['type']
    net_name = config.net_name

    if model_type.lower() == 'dssm':
        model = engine.load_model(pr_dir, net_name)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    query = click.prompt("What do you want to search?", type=str)
    while query and query != 'exit':
        results = model_predict(ctx.obj['CONFIG'],
                                model,
                                query,
                                num_largest)
        for res in results:
            print(res)
        query = click.prompt("What do you want to search?", type=str)


@cli.command()
@click.pass_context
def search(ctx):
    from officeanswers.search import build_search_index

    logger.info("Build search index...")

    config = ctx.obj['CONFIG']
    preprocess_dir = config.paths['preprocess_dir']
    embed_model = get_inference_model(config)

    pre = engine.load_preprocessor(preprocess_dir,
                                   config.net_name)

    docs, embeds = build_document_embeddings(config)
    search = build_search_index(embeds)

    query = click.prompt("\nWhat do you want to search?\n", type=str)
    while query and query != 'exit':
        sparse_input = pre.transform_list([query])[0]
        sparse_input = np.expand_dims(sparse_input, axis=0)
        dense_input = embed_model.predict(sparse_input)[0]

        print(type(dense_input))
        print(dense_input.shape)

        idxs, dists = search.knnQuery(dense_input, k=3)
        scores = list(zip(idxs, dists))
        scores.sort(key=lambda x: x[1], reverse=True)

        for idx, dist in scores:
            print(f'\nCosine Dist: {dist:.4f}\n---------------\n', docs[idx])

        query = click.prompt("\nWhat do you want to search?\n", type=str)


@cli.command()
@click.pass_context
def search_universal(ctx):
    import nmslib
    import tensorflow as tf
    import tensorflow_hub as hub

    from officeanswers.search import build_search_index

    logger.info("Download universal sentence encoder...")

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    embed = hub.Module(module_url)

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    dataset_path = ctx.obj['CONFIG'].inputs['share']['custom_corpus']
    index = 'custom_index'

    docs = []
    with open(dataset_path, 'r') as f:
        for line in f:
            line = line.strip()
            try:
                question, doc, label = line.split('\t')
            except ValueError:
                error_msg = "Invalid format for relation text." + \
                    "Should be `question\tdocument\tlabel`\n" + f"{line}"
                raise ValueError(error_msg)
            docs.append(doc)

    if not os.path.exists(index):
        with tf.Session(config=config) as session:
            session.run([tf.global_variables_initializer(),
                         tf.tables_initializer()])
            embeds = session.run(embed(docs))

        search = build_search_index(embeds)
        search.saveIndex(index)
    else:
        search = nmslib.init(method='hnsw', space='cosinesimil')
        search.loadIndex(index)

    with tf.Session(config=config) as session:
        session.run([tf.global_variables_initializer(),
                     tf.tables_initializer()])
        text = tf.placeholder(dtype=tf.string, shape=[None])
        embed_query = embed(text)

        query = click.prompt("What do you want to search?", type=str)
        while query and query != 'exit':
            dense_input = session.run(embed_query, feed_dict={text: [query]})
            idxs, dists = search.knnQuery(dense_input[0])

            for idx, dist in zip(idxs, dists):
                print(
                    f'\nCosine dist:{dist:.4f}\n---------------\n',
                    docs[idx])
            query = click.prompt("What do you want to search?", type=str)


@cli.command()
@click.pass_context
def search_tdif(ctx):
    from gensim.corpora import Dictionary
    from gensim.models import TfidfModel
    from gensim.similarities import Similarity
    from nltk.tokenize import word_tokenize

    logger.info("Build search index...")

    config = ctx.obj['CONFIG']

    dataset_path = config.inputs['share']['custom_corpus']
    raw_docs = []
    with open(dataset_path, 'r') as f:
        for line in f:
            line = line.strip()
            try:
                question, doc, label = line.split('\t')
            except ValueError:
                error_msg = "Invalid format for relation text." + \
                    "Should be `question\tdocument\tlabel`"
                logger.error(error_msg)
                raise
            raw_docs.append(doc)

    docs = [[w.lower() for w in word_tokenize(text)] for text in raw_docs]
    dictionary = Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    tf_idf = TfidfModel(corpus)
    sims = Similarity('tf_idf', tf_idf[corpus],
                      num_features=len(dictionary))

    query = click.prompt("\nWhat do you want to search?\n", type=str)
    while query and query != 'exit':
        dense_input = [w.lower() for w in word_tokenize(query)]
        dense_input = dictionary.doc2bow(dense_input)
        dense_input = tf_idf[dense_input]

        res = sims[dense_input]
        res = sorted(enumerate(res), key=lambda item: -item[1])
        for i in range(3):
            idx, dist = res[i]
            print(f'\nCosine Dist: {dist:.4f}\n---------------\n',
                  raw_docs[idx])

        query = click.prompt("\nWhat do you want to search?\n", type=str)


@cli.command()
@click.pass_context
def test(ctx):
    logger.info('Testing model...')
    model_evaluate(ctx.obj['CONFIG'])


if __name__ == "__main__":
    cli(obj={})
