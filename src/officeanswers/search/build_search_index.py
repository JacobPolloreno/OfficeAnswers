import nmslib
import numpy as np

from ..preprocess import Hasher


def build_search_index(embeddings: np.ndarray):
    search_index = nmslib.init(method='hnsw', space='cosinesimil')
    search_index.addDataPointBatch(embeddings)
    search_index.createIndex({'post': 2}, print_progress=True)
    return search_index
