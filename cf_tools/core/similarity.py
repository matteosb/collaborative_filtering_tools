import numpy as np


def cosine(v1, v2):
    return np.dot(v1, v2) / (_norm(v1) * _norm(v2))


def _norm(v):
    return np.sqrt(np.power(v, 2).sum())
