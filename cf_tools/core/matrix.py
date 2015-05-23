"""Core matrix and vector operations"""
from .models import IndexBiMap
import numpy as np


def build_matrix_from_pref_vectors(pref_vectors, score_cb=None):
    """Build a user-item matrix from a list of preference vectors

    Users are rows, items are columns. This convention is used in all
    other functions in this package.

    Arguments:
      pref_vectors: list[PreferenceVector], a colelction of preference vectors,
        note that the data structure is iterated over twice so some collections
        won't work
      score_cb: numeric -> numeric, an optional callback to modify scores

    Returns:
      A (user-item numpy matrix, user IndexBiMap, item IndexBiMap) 3-tuple
    """
    if score_cb is None:
        score_cb = lambda x: x
    user_idxs = IndexBiMap()
    item_idxs = IndexBiMap()
    for vector in pref_vectors:
        user_idxs.get_or_issue_index(vector.user_id)
        for item, _ in vector.iter_prefs():
            item_idxs.get_or_issue_index(item)
    # We loop through the data twice here but build a dense matrix
    # the alternatives are:
    # 1) loop once, build a sparse matrix
    #     and convert to dense (numpy likes dense matrices)
    # 2) if dimensions are known, build the zero matrix first,
    #     loop once and fill in
    mat = np.zeros((user_idxs.size(), item_idxs.size()), dtype=np.float32)
    for vector in pref_vectors:
        row = user_idxs.get_index(vector.user_id)
        for item, score in vector.iter_prefs():
            column = item_idxs.get_index(item)
            mat[row][column] = score_cb(score)
    return np.matrix(mat), user_idxs, item_idxs


def fast_cosine_similarity_matrix(mat):
    """A very fast way to create a cosine similarity matrix

    This function is fast because it performs all operations
    as matrix operations and, where possible, in-place. Some
    cursory testing showed at least an of magnitude improvement
    over calculating the cosine between vectors individually using
    numpy.

    Arguments:
      mat: an n x m numpy matrix

    Returns:
      A n x n matrix where the cell (i,j) is the cosine similarity
      between the ith and jth row vectors.
    """
    # calculate the inner products between all row vectors
    products = mat * mat.T
    # for each row vector, the magnitude can be read off of the
    # resulting diagonal
    norms = np.sqrt(np.diag(products))
    # for each cell, calculate the product of the norms, this is the denominator
    # for the cosine calculation
    denominators = np.outer(norms, norms)
    # divide in place
    np.divide(products, denominators, out=products)
    return products


def item_to_item_cos_sim_mat(mat):
    """Generate a item-to-item similarity matrix, using cosine similarity"""
    return fast_cosine_similarity_matrix(mat.T)


def user_to_user_cos_sim_mat(mat):
    """Generate a user-to-user similarity matrix, using cosine similarity"""
    return fast_cosine_similarity_matrix(mat)


def top_n_similar_items(item_id, item_sim_mat, item_idxs, n=10):
    item_idx = item_idxs.get_index(item_id)
    top_n_idxs = _top_n_plus_one(item_sim_mat[item_idx, :], n)
    return [item_idxs.get_id(idx) for idx in top_n_idxs if idx != item_idx]


def get_user_vector_from_user_item_mat(id_, user_item_mat, user_idxs):
    return user_item_mat[user_idxs.get_index(id_), :].T


def _top_n_plus_one(_1d_mat, n):
    return _1d_mat.A1.argsort()[-1:-(n + 2):-1].tolist()

