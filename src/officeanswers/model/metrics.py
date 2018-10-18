"""Metrics for training information retrieval systems

    source:
    https://github.com/faneshion/MatchZoo/blob/master/matchzoo/metrics/evaluations.py
"""
import logging
import math
import numpy as np
import random

logger = logging.getLogger(__name__)


def _to_list(x):
    if isinstance(x, list):
        return x
    return [x]


def mean_average_precision(y_true, y_pred, rel_threshold=0.0):
    s = 0.
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    c = list(zip(y_true, y_pred))
    random.shuffle(c)
    c = sorted(c, key=lambda x: x[1], reverse=True)
    ipos = 0
    for j, (g, p) in enumerate(c):
        if g > rel_threshold:
            ipos += 1.
            s += ipos / (j + 1.)
    if ipos == 0:
        s = 0.
    else:
        s /= ipos
    return s


def mean_reciprocal_rank(y_true, y_pred):
    s = 0.
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    c = list(zip(y_true, y_pred))
    return s


def ndcg(k=10):
    def top_k(y_true, y_pred, rel_threshold=0.):
        if k <= 0.:
            return 0.
        y_true = _to_list(np.squeeze(y_true).tolist())
        y_pred = _to_list(np.squeeze(y_pred).tolist())
        c = list(zip(y_true, y_pred))
        random.shuffle(c)
        c_g = sorted(c, key=lambda x: x[0], reverse=True)
        c_p = sorted(c, key=lambda x: x[1], reverse=True)
        idcg = 0.
        ndcg = 0.
        for i, (g, p) in enumerate(c_g):
            if i >= k:
                break
            if g > rel_threshold:
                idcg += (math.pow(2., g) - 1.) / math.log(2. + i)
        for i, (g, p) in enumerate(c_p):
            if i >= k:
                break
            if g > rel_threshold:
                ndcg += (math.pow(2., g) - 1.) / math.log(2. + i)
        if idcg == 0.:
            return 0.
        else:
            return ndcg / idcg
    return top_k


def precision(k=10):
    def top_k(y_true, y_pred, rel_threshold=0.):
        if k <= 0:
            return 0.
        y_true = _to_list(np.squeeze(y_true).tolist())
        y_pred = _to_list(np.squeeze(y_pred).tolist())
        c = list(zip(y_true, y_pred))
        random.shuffle(c)
        c = sorted(c, key=lambda x: x[1], reverse=True)
        prec = 0.
        for i, (g, p) in enumerate(c):
            if i >= k:
                break
            if g > rel_threshold:
                prec += 1
        prec /= k
        return prec
    return top_k
