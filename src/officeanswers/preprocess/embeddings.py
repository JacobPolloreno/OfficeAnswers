import os
import logging
import numpy as np
import typing

from ..util import Config

from keras.models import load_model
from matchzoo import engine
from matchzoo import losses

logger = logging.Logger(__name__)


def build_document_embeddings(config: Config) -> typing.Tuple[typing.List[str],
                                                              np.ndarray]:
    """
    Build document embeddings by running inference on model

    Args:
        config (Config): configuration model object after preprocessing and
            training phase

    Return:
        tuple with raw docs and their numpy embeddings
    """
    logger.info("Building embeddings...")
    try:
        dataset_path = config.inputs['share']['custom_corpus']
        preprocess_dir = config.paths['preprocess_dir']
        processed_dir = config.paths['processed_dir']
    except KeyError as e:
        error_msg = f"KeyError {e}\nCheck config file"
        logger.error(error_msg)
        raise KeyError(e)

    if not os.path.exists(dataset_path):
        error_msg = f"Dataset: `{dataset_path}` does not exist."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    model_path = os.path.join(processed_dir,
                              f"{config.net_name}.h5")
    if not os.path.exists(model_path):
        error_msg = f"{model_path} does not exist. Train model first"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info("Loading embedding model...")
    custom_object = {}
    custom_object['rank_hinge_loss'] = losses.rank_hinge_loss
    embed_model = load_model(model_path,
                             custom_objects=custom_object).get_layer('model_1')
    docs = []
    embeddings = []
    logger.info("Getting embeddings...")
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
            docs.append(doc)

    pre = engine.load_preprocessor(preprocess_dir,
                                   config.net_name)

    preprocessed_docs = pre.transform_list(docs)
    for doc in preprocessed_docs:
        sparse_input = np.expand_dims(doc, axis=0)
        embeddings.append(embed_model.predict(sparse_input)[0])

    return docs, np.array(embeddings)
