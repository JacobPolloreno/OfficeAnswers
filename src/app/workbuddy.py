import config
import hug
import logging
import nmslib
import numpy as np
import os
import sys

from matchzoo import engine

file_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(file_dir, '..'))
sys.path.append(base_dir)

from officeanswers.preprocess import build_document_embeddings
from officeanswers.model import get_inference_model
from officeanswers.search import build_search_index
from officeanswers.util import Config

logger = logging.getLogger(__file__)

logger.info("Build search index...")

config_path = config.OFFICEANSWERS_CONFIG
if not config_path or not os.path.exists(config_path):
    raise IOError("Config file not or envionmental variable not set." +
                  "Make sure OA_CONFIG is set to config file path" +
                  f"Path supplied {config_path}")

base_dir = config.OFFICEANSWERS_DIR

model_config = Config()
model_config.from_json_file(config_path)
data_dir = os.path.join(base_dir, 'data')
preprocess_dir = os.path.join(data_dir,
                              'preprocessed')
processed_dir = os.path.join(data_dir,
                             'processed')
model_config.paths['preprocess_dir'] = preprocess_dir
model_config.paths['processed_dir'] = processed_dir

embed_model = get_inference_model(config)
if 'preprocess' in model_config.inputs['share']:
    pre = engine.load_preprocessor(preprocess_dir,
                                   model_config.inputs['share']['preprocess'])
else:
    pre = engine.load_preprocessor(preprocess_dir,
                                   model_config.net_name)

model_config.inputs['share']['custom_corpus'] = os.path.join(
    base_dir,
    model_config.inputs['share']['custom_corpus'])
docs, embeds = build_document_embeddings(config)

logger.info("Loading search index...")
index_name = 'custom_index'
if not os.path.exists(index_name):
    logger.info("Search index not found. Building it...")
    search_engine = build_search_index(embeds)
    search_engine.saveIndex(index_name)
else:
    search_engine = nmslib.init(method='hnsw', space='cosinesimil')
    search_engine.loadIndex(index_name)

logger.info("Model ready to query.")


@hug.cli()
@hug.get(examples='query=how%20to%20connect%20to%20printer')
@hug.local()
def search(query: hug.types.text):
    sparse_input = pre.transform_list([query])[0]
    sparse_input = np.expand_dims(sparse_input, axis=0)
    dense_input = embed_model.predict(sparse_input)[0]

    idxs, dists = search_engine.knnQuery(dense_input, k=3)
    res = []
    for idx, dist in zip(idxs, dists):
        res.append((dist, docs[idx]))
    res.sort(key=lambda x: x[0], reverse=True)

    output = {}
    for k, v in enumerate(res):
        dist, doc = v
        output[str(k)] = {'dist': str(dist), 'doc': doc}

    return {'results': output}


if __name__ == '__main__':
    search.interface.cli()
