import dill
import logging
import os
import typing

from ..ingestion import DSSMPrepare
from ..ingestion import DSSMFormatter
from .preprocessor import DSSMUEPreprocessor
from ..util import Config

from matchzoo import preprocessor

logger = logging.getLogger(__name__)


def prepare(config: Config,
            save_relations: bool=False) -> typing.Tuple:
    """
    Prepare tab-seperated corpus that's been formatted properly
        with DSSMFormatter for preprocessing
    """
    logger.info('Preparing data model...')

    pp_dir = config.paths['preprocess_dir']

    if not os.path.exists(pp_dir):
        os.mkdir(pp_dir)
    elif not os.path.isdir(pp_dir):
        error_msg = "Data path already exists and is a file." + \
            "Should be a directory, [{pp_dir}]"
        logger.error(error_msg)
        raise FileExistsError(error_msg)

    model_type = config.model['type']
    if model_type.lower() == 'dssm' or model_type.lower() == 'dssm_ue':
        prep = DSSMPrepare()
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    try:
        corpus_path = config.inputs['share']['raw_corpus']
    except KeyError as e:
        logger.error(
            "Config file doesn't have corpus path in inputs.share.corpus")
        raise

    if not os.path.exists(corpus_path):
        error_msg = f"{corpus_path} does not exists.\n" + \
            "Run prepare_wikiqa.py in data directory."
        logger.info(error_msg)
        raise FileExistsError(error_msg)

    try:
        custom_path = config.inputs['share']['custom_corpus']
    except KeyError as e:
        pass

    if custom_path and not os.path.exists(custom_path):
        error_msg = f"{custom_path} does not exists.\n" + \
            "Check configuration file"
        logger.info(error_msg)
        raise FileExistsError(error_msg)

    if not custom_path or custom_path == corpus_path:
        corpus_q, corpus_d, rels = prep.from_one_corpus(corpus_path)
    else:
        corpus_q, corpus_d, rels = prep.from_corpus([corpus_path,
                                                     custom_path])

    logger.info(f"total questions: {len(corpus_q)}")
    logger.info(f"total documents: {len(corpus_d)}")
    logger.info(f"total relations: {len(rels)}")

    if not len(corpus_q) or not len(corpus_d):
        error_msg = f"{corpus_path} is empty."
        logger.info(error_msg)
        raise IOError(error_msg)

    rel_train, rel_valid, rel_test = prep.split_train_valid_test(rels)
    if save_relations:
        relations_files = ['relations.txt', 'rel_train.txt',
                           'rel_valid.txt', 'rel_test.txt']
        relations_files = [f"{config.net_name}_{p}" for p in relations_files]
        relations_paths = [os.path.join(pp_dir, rf)
                           for rf in relations_files]
        for path, data in zip(relations_paths, [rels, rel_train, rel_valid,
                                                rel_test]):
            prep.save_relations(path, data)

        logger.info(f"Saved relations splits to {pp_dir}")

    logger.info("Done formatting data...")

    relations_dict = {
        "train": rel_train,
        "test": rel_test,
        "valid": rel_valid}

    return corpus_q, corpus_d, relations_dict


def preprocess(config: Config,
               corpus_d: typing.Dict[str, str],
               corpus_q: typing.Dict[str, str],
               relations: typing.Dict[str,
                                      typing.List[typing.Tuple[str, str, str]]]) -> None:
    logger.info('Preprocessing data...')

    net_name = config.net_name
    pp_dir = config.paths['preprocess_dir']

    model_type = config.model['type']
    if model_type.lower() == 'dssm':
        formatter = DSSMFormatter()
        pre = preprocessor.DSSMPreprocessor()
    elif model_type.lower() == 'dssm_ue':
        formatter = DSSMFormatter()
        pre = DSSMUEPreprocessor()
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    inputs = {'questions': corpus_q, 'documents': corpus_d}

    logger.info("Transforming and saving train data...")
    train = formatter.from_inputs(inputs, relations['train'], stage='train')
    logger.info(f"Size of train data {len(train)}...")
    pre = pre.fit(train.values)
    processed_train = pre.transform(train.values, stage='train')
    processed_train.DATA_FILENAME = net_name + '_train'
    processed_train.save(dirpath=pp_dir,
                         name=net_name + "_train")
    del train
    del processed_train

    logger.info("Transforming and saving validation data...")
    val = formatter.from_inputs(inputs, relations['valid'], stage='train')
    logger.info(f"Size of valid data {len(val)}...")
    processed_val = pre.transform(val.values,
                                  stage='test',
                                  cache=False)
    processed_val.DATA_FILENAME = net_name + '_valid'
    processed_val.save(dirpath=pp_dir,
                       name=net_name + "_valid")

    del val
    del processed_val

    logger.info("Transforming and saving test data...")
    test = formatter.from_inputs(inputs, relations['test'],
                                 stage='train')
    logger.info(f"Size of test data {len(test)}...")
    processed_test = pre.transform(test.values,
                                   stage='test',
                                   cache=False)
    processed_test.DATA_FILENAME = net_name + '_test'
    processed_test.save(dirpath=pp_dir,
                        name=net_name + "_test")

    del test
    del processed_test

    pre.save(dirpath=pp_dir,
             name=config.net_name)

    logger.info('Saving corpus questions and documents...')
    corpus_d_path = os.path.join(pp_dir,
                                 net_name + "_documents.dill")
    dill.dump(corpus_d, open(corpus_d_path, 'wb'))

    corpus_q_path = os.path.join(pp_dir,
                                 net_name + "_questions.dill")
    dill.dump(corpus_q, open(corpus_q_path, 'wb'))


def prepare_and_preprocess(config: Config,
                           save_relations: bool=False) -> None:
    corpus_q, corpus_d, relations = prepare(config, save_relations)
    preprocess(config, corpus_d, corpus_q, relations)

    logger.info(f"{len(relations['train'])} train relations")
    logger.info(f"{len(relations['valid'])} val relations")
    logger.info(f"{len(relations['test'])} test relations")
