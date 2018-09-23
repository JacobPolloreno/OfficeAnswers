"""Office Answers CLI"""

import click
import dill
import heapq
import logging
import pandas as pd
import os

from ingestion import DSSMPrepare
from ingestion import DSSMFormatter
from util import Config

from matchzoo import engine
from matchzoo import generators
from matchzoo import datapack
from matchzoo import models
from matchzoo import preprocessor
from matchzoo import tasks

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
def prepare_and_preprocess(ctx, save: bool=False):
    click.echo('Preparing data model...')

    data_dir = os.path.abspath('./data/preprocessed')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    elif not os.path.isdir(data_dir):
        error_msg = "Data path already exists and is a file." + \
            "Should be a directory"
        logger.error(error_msg)
        raise FileExistsError(error_msg)

    cfg = ctx.obj['CONFIG']
    model_type = cfg.model['model_py']

    if model_type.lower() == 'dssm':
        prep = DSSMPrepare()
        formatter = DSSMFormatter()
        pre = preprocessor.DSSMPreprocessor()
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    try:
        corpus_q, corpus_d, rels = prep.from_one_corpus(
            cfg.inputs['share']['raw_corpus'])

        logger.info(f"total questions: {len(corpus_q)}")
        logger.info(f"total documents: {len(corpus_d)}")
        logger.info(f"total relations: {len(rels)}")

        rel_train, rel_valid, rel_test = prep.split_train_valid_test(rels)
        if save:
            relations_files = ['relation_train.txt', 'relation_valid.txt',
                               'relation_test.txt']
            relations_paths = [os.path.join(data_dir, rf)
                               for rf in relations_files]
            for path, data in zip(relations_paths, [rel_train, rel_valid,
                                                    rel_test]):
                prep.save_relations(path, data)

        logger.info(f"Saved relations splits to {data_dir}")

        click.echo('Preprocessing data...')

        inputs = {'questions': corpus_q, 'documents': corpus_d}
        train = formatter.from_inputs(inputs, rel_train, stage='train')
        test = formatter.from_inputs(inputs, rel_test, stage='train')
        val = formatter.from_inputs(inputs, rel_valid, stage='train')

        processed_train = pre.fit_transform(train.values, stage='train')
        processed_val = pre.transform(val.values, stage='train')
        processed_test = pre.transform(test.values, stage='train')

        click.echo('Saving preprocessed data...')

        processed_train.save(dirpath=data_dir, filename='train.dill')
        processed_test.save(dirpath=data_dir, filename='test.dill')
        processed_val.save(dirpath=data_dir, filename='val.dill')

        pre.save(dirpath=data_dir)

        click.echo('Saving corpus questions and documents...')
        corpus_d_path = os.path.join(data_dir, 'documents.dill')
        dill.dump(corpus_d, open(corpus_d_path, 'wb'))
        corpus_q_path = os.path.join(data_dir, 'questions.dill')
        dill.dump(corpus_d, open(corpus_q_path, 'wb'))

    except KeyError:
        logger.error(
            "Config file doesn't have corpus path in inputs.share.corpus")


@cli.command()
@click.pass_context
def train(ctx):
    click.echo('Loading preprocessed data...')

    cfg = ctx.obj['CONFIG']
    data_dir = os.path.abspath('./data')
    pre_dir = os.path.join(data_dir, 'preprocessed')

    processed_train = datapack.load_datapack(pre_dir, 'train.dill')
    processed_val = datapack.load_datapack(pre_dir, 'val.dill')

    task = tasks.Ranking()
    input_shapes = processed_train.context['input_shapes']
    generator_train = generators.PointGenerator(processed_train,
                                                task,
                                                stage='train')
    processed_val = datapack.load_datapack(pre_dir, 'val.dill')
    generator_val = generators.PointGenerator(processed_val,
                                              task,
                                              stage='train')

    model_type = cfg.model['model_py']

    click.echo('Loading model...')
    if model_type.lower() == 'dssm':
        model = models.DSSMModel()
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    model.params['input_shapes'] = input_shapes
    model.params['task'] = task
    model.guess_and_fill_missing_params()
    click.echo(f"Model parameters:\n {model.params}")

    click.echo('Build model...')
    model.build()
    model.compile()
    logger.info(model._backend.summary())

    click.echo('Training model...')
    model.fit_generator(generator_train,
                        steps_per_epoch=200,
                        val_generator=generator_val,
                        epochs=3,
                        verbose=1)

    click.echo('Saved model...')
    try:
        model.save(os.path.join(data_dir, 'processed'))
    except FileExistsError:
        logger.error("File exists already.")

    click.echo('\nPredict...')
    processed_test = datapack.load_datapack(pre_dir, 'test.dill')
    generator_test = generators.PointGenerator(processed_test,
                                               task,
                                               stage='train')
    outputs = model.evaluate_generator(generator_test)
    if outputs and len(outputs):
        loss, metrics = outputs[0], outputs[1:]
        metric_labels = model.params['metrics']
        click.echo(f"Test loss: {loss}")
        for name, out in zip(metric_labels, metrics):
            click.echo(f"{name}: {out}")


@cli.command()
@click.pass_context
def predict(ctx):
    click.echo('Running predictions...')

    cfg = ctx.obj['CONFIG']
    data_dir = os.path.abspath('./data')
    pre_dir = os.path.join(data_dir, 'preprocessed')

    model_type = cfg.model['model_py']
    corpus_d_path = os.path.join(pre_dir, 'documents.dill')
    model_path = os.path.join(data_dir, 'processed')

    docs = dill.load(open(corpus_d_path, 'rb'))
    doc_lookup = list(docs.keys())
    num_docs = len(doc_lookup)
    docs_df = pd.DataFrame.from_dict(docs,
                                     orient='index',
                                     columns=['Document'])
    docs_df['QID'] = 'Q'

    task = tasks.Ranking()
    pre = engine.load_preprocessor(dirpath=pre_dir)
    click.echo('Loading model...')
    if model_type.lower() == 'dssm':
        model = engine.load_model(model_path)
        query = click.prompt("What do you want to search?", type=str)
        while query and query != 'exit':
            query_df = docs_df.copy()
            query_df['Question'] = query
            inputs = pre.transform(list(query_df.itertuples()),
                                   stage='predict')
            gen_predict = generators.PointGenerator(inputs,
                                                    task,
                                                    shuffle=False,
                                                    stage='test')
            predictions = model._backend.predict_generator(gen_predict,
                                                           verbose=1)
            idx = heapq.nlargest(5, range(num_docs),
                                 predictions.ravel().take)
            for candidate in idx:
                did = doc_lookup[candidate]
                d = docs[did]
                score = predictions[candidate][0]
                click.echo(f"{did} [{score}]: {d}")
            query = click.prompt("What do you want to search?", type=str)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")


@cli.command()
@click.pass_context
def test(ctx):
    click.echo('Testing model...')


if __name__ == "__main__":
    cli(obj={})
