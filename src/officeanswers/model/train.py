import logging
import os

from .metrics import mean_average_precision
from .metrics import ndcg
from ..util import Config

from matchzoo import datapack
from matchzoo import generators
from matchzoo import tasks
from matchzoo import losses
from matchzoo import models

logger = logging.getLogger(__name__)


def train(config: Config) -> None:
    """
    Load preprocessed data and preprocess function to train
    and save model.

    Args:
        config (Config): model config object
    """
    logger.info('Running training process...')

    net_name = config.net_name
    pp_dir = config.paths['preprocess_dir']
    pr_dir = config.paths['processed_dir']

    logger.info('Loading preprocessed data...')
    if 'preprocess' in config.inputs['share']:
        pr_name = config.inputs['share']['preprocess']
        processed_train = datapack.load_datapack(pp_dir,
                                                 name=pr_name + "_train")
        processed_val = datapack.load_datapack(pp_dir,
                                               name=pr_name + "_valid")
    else:
        processed_train = datapack.load_datapack(pp_dir,
                                                 name=net_name + "_train")
        processed_val = datapack.load_datapack(pp_dir,
                                               name=net_name + "_valid")

    task = tasks.Ranking()
    input_shapes = processed_train.context['input_shapes']

    logger.info('Creating generators from preprocessed data...')
    generator_train = generators.PairGenerator(
        processed_train,
        batch_size=config.inputs['train']['batch_size'],
        stage='train')
    del processed_train
    generator_val = generators.ListGenerator(processed_val)
    del processed_val

    logger.info('Loading model...')
    model_type = config.model['type']
    if model_type.lower() == 'dssm':
        model = models.DSSMModel()
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    model.params['input_shapes'] = input_shapes
    model.params['task'] = task
    model.params['loss'] = [losses.rank_hinge_loss]
    model.params['metrics'] = []
    model.guess_and_fill_missing_params()
    logger.info(f"Model parameters:\n {model.params}")

    logger.info('Build model...')
    model.build()
    model.compile()
    logger.info(model._backend.summary())

    train_params = config.inputs['train']
    save_weights_iters = train_params['save_weights_iter']
    weights_file = f"{net_name}.weights.%d"

    logger.info('Training model...')
    verbose = train_params['verbose'] if train_params['verbose'] else 1
    for epoch in range(1, train_params['epochs'] + 1):
        model.fit_generator(
            generator_train,
            steps_per_epoch=train_params['steps_per_epoch'],
            epochs=1,
            verbose=verbose)

        res = {}
        res['MAP'] = 0.0
        res['NCDG@3'] = 0.0
        res['NCDG@5'] = 0.0
        num_valid = 0
        for i in range(len(generator_val)):
            input_data, y_true = generator_val[i]
            y_pred = model.predict(input_data,
                                   batch_size=len(y_true),
                                   verbose=0)
            res['MAP'] += mean_average_precision(y_true, y_pred)
            res['NCDG@3'] += ndcg(3)(y_true, y_pred)
            res['NCDG@5'] += ndcg(5)(y_true, y_pred)
            num_valid += 1

        generator_val.reset()
        res['VAL LOSS'] = model._backend.evaluate_generator(
            generator_val,
            verbose=0)
        logger.info(f"Iter: {epoch} / {train_params['epochs'] + 1}\t" +
                    '\t'.join(
                        [f"{k}={v / num_valid:.3f}" for k, v in res.items()]))

        if (epoch + 1) % save_weights_iters == 0:
            model._backend.save_weights(
                os.path.join(pr_dir, weights_file % (epoch + 1)))
            logger.info(f"Saved model at iter {epoch}")

    logger.info('Saving model...')
    try:
        model.save(pr_dir, config.net_name)
    except FileExistsError:
        logger.error("File exists already.")
