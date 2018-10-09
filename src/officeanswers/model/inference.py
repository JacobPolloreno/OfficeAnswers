import os
import logging

from ..util import Config

from matchzoo import engine
from matchzoo import losses
from keras.models import load_model
from keras.models import Model

logger = logging.getLogger(__name__)


def get_inference_model(config: Config) -> Model:
    """
    Extract and return trained DSSM specific MLP that's
    used to create embeddings.

    Returns:
        keras model with pretrained weights

    Usage:
        >>> embed_model = get_inference_model(config)
        >>> sparse_input = pre.transform_list([query])[0]
        >>> sparse_input = np.expand_dims(sparse_input, axis=0)
        >>> dense_input = embed_model.predict(sparse_input)[0]
    """
    logger.info('Getting inference model for embeds...')

    net_name = config.net_name
    pr_dir = config.paths['processed_dir']
    model_path = os.path.join(pr_dir,
                              f"{config.net_name}.h5")

    custom_object = {}
    custom_object['rank_hinge_loss'] = losses.rank_hinge_loss

    logger.info('Loading model...')
    model = load_model(model_path,
                       custom_objects=custom_object)
    embed_model = model.get_layer('model_1')
    return embed_model
