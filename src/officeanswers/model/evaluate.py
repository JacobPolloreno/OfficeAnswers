import logging
import os
import typing

from .metrics import mean_average_precision
from .metrics import ndcg
from ..util import Config

from keras.models import load_model
from matchzoo import datapack
from matchzoo import generators
from matchzoo import losses

logger = logging.getLogger(__name__)


def evaluate(config: Config) -> typing.Dict[str, float]:
    """
    Load and evaluate model on a list generator

    Return:
        dict of metrics for the model run
    """
    logger.info('Running evaluation process...')

    net_name = config.net_name
    pp_dir = config.paths['preprocess_dir']
    pr_dir = config.paths['processed_dir']
    model_path = os.path.join(pr_dir,
                              f"{config.net_name}.h5")

    logger.info('Loading model...')
    custom_object = {}
    custom_object['rank_hinge_loss'] = losses.rank_hinge_loss
    model = load_model(model_path,
                       custom_objects=custom_object)

    logger.info('Loading preprocessed test data...')
    processed_test = datapack.load_datapack(pp_dir,
                                            name=net_name + "_test")
    generator_test = generators.ListGenerator(processed_test,
                                              stage='train')
    res = {}
    res['MAP'] = 0.0
    res['NCDG@3'] = 0.0
    res['NCDG@5'] = 0.0
    num_valid = 0

    logger.info('Evaluating model...')
    for i in range(len(generator_test)):
        input_data, y_true = generator_test[i]
        y_pred = model.predict(input_data,
                               batch_size=len(y_true),
                               verbose=0)
        res['MAP'] += mean_average_precision(y_true, y_pred)
        res['NCDG@3'] += ndcg(3)(y_true, y_pred)
        res['NCDG@5'] += ndcg(5)(y_true, y_pred)
        num_valid += 1

    logger.info('\t'.join(
        [f"{k}={v / num_valid:.3f}" for k, v in res.items()]))
    return res
