import logging
import typing

from ..util import Config

from matchzoo import datapack
from matchzoo import generators
from matchzoo import tasks
from matchzoo import models

logger = logging.getLogger(__name__)


def train(config: Config) -> typing.List[typing.Tuple[str, float]]:
    logger.info('Loading preprocessed data...')

    net_name = config.net_name
    pp_dir = config.paths['preprocess_dir']
    pr_dir = config.paths['processed_dir']

    processed_train = datapack.load_datapack(pp_dir,
                                             name=net_name + "_train")
    processed_val = datapack.load_datapack(pp_dir,
                                           name=net_name + "_valid")

    task = tasks.Ranking()
    input_shapes = processed_train.context['input_shapes']
    generator_train = generators.PointGenerator(processed_train,
                                                task,
                                                stage='train')
    generator_val = generators.PointGenerator(processed_val,
                                              task,
                                              stage='train')

    model_type = config.model['type']

    logger.info('Loading model...')
    if model_type.lower() == 'dssm':
        model = models.DSSMModel()
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    model.params['input_shapes'] = input_shapes
    model.params['task'] = task
    model.guess_and_fill_missing_params()
    logger.info(f"Model parameters:\n {model.params}")

    logger.info('Build model...')
    model.build()
    model.compile()
    logger.info(model._backend.summary())

    logger.info('Training model...')
    train_params = config.inputs['train']
    model.fit_generator(generator_train,
                        steps_per_epoch=train_params['steps_per_epoch'],
                        val_generator=generator_val,
                        epochs=train_params['epochs'],
                        verbose=train_params['verbose'])

    logger.info('Saved model...')
    try:
        model.save(pr_dir, config.net_name)
    except FileExistsError:
        logger.error("File exists already.")

    logger.info('\nPredict...')
    processed_test = datapack.load_datapack(pp_dir,
                                            name=net_name + "_test")
    generator_test = generators.PointGenerator(processed_test,
                                               task,
                                               stage='train')
    outputs = model.evaluate_generator(generator_test)
    results = []
    if outputs and len(outputs):
        loss, metrics = outputs[0], outputs[1:]
        metric_labels = model.params['metrics']
        logger.info(f"Test loss: {loss}")
        for name, out in zip(metric_labels, metrics):
            logger.info(f"{name}: {out}")
            results.append((name, out))
    return results
