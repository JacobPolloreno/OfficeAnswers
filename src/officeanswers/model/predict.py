import dill
import heapq
import logging
import pandas as pd
import typing
import os

from officeanswers.util import Config

from matchzoo import engine
from matchzoo import generators
from matchzoo import tasks

logger = logging.getLogger(__name__)


def predict(config: Config,
            model: engine.BaseModel,
            query: str,
            nlargest: int=5) -> typing.List[typing.Tuple[str, float, str]]:
    logger.info('Running predictions...')

    net_name = config.net_name
    pp_dir = config.paths['preprocess_dir']
    corpus_d_path = os.path.join(pp_dir,
                                 net_name + "_documents.dill")

    docs = dill.load(open(corpus_d_path, 'rb'))
    doc_lookup = list(docs.keys())
    num_docs = len(doc_lookup)
    docs_df = pd.DataFrame.from_dict(docs,
                                     orient='index',
                                     columns=['Document'])
    docs_df['QID'] = 'Q'
    task = tasks.Ranking()
    pre = engine.load_preprocessor(dirpath=pp_dir,
                                   name=net_name)

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
    idx = heapq.nlargest(nlargest, range(num_docs),
                         predictions.ravel().take)
    results = []
    for candidate in idx:
        did = doc_lookup[candidate]
        d = docs[did]
        score = predictions[candidate][0]
        results.append((did, score, d))

    return results
