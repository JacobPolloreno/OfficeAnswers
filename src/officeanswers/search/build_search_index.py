import nmslib
import numpy as np


def build_search_index(embeddings: np.ndarray):
    """
    Build a search index for NMSLIB semantic search library

    Args:
        embeddings (numpy array) : array containing precomputed
            embeddings that you want to search

    Return:
        a nmslib search index that can be cached for later or
        used to perform searches


    Usage:
        >>> docs, embeds = build_document_embeddings(config)
        >>> search = build_search_index(embeds)
        >>> ...
        >>> idxs, dists = search.knnQuery(dense_input, k=5)
    """
    search_index = nmslib.init(method='hnsw', space='cosinesimil')
    search_index.addDataPointBatch(embeddings)
    search_index.createIndex({'post': 2}, print_progress=True)
    return search_index
