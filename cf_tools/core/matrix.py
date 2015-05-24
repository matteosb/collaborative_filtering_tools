"""Core matrix and vector operations and classes"""
import numpy as np


class IndexBiMap(object):
    """A bidirectional map used to issue and track matrix indices.

    Assists creating dense matrecies by converting arbitrary ids to indecies
    and indecies back to ids.
    When creating a matrix, it is typical to use to bimaps, one for the rows
    and one for the columns.
    """

    def __init__(self):
        self.id_to_index = {}
        self.index_to_id = {}
        self.curr_index = -1

    def get_index(self, the_id):
        """Get the index for an id.

        Raises:
          KeyError if the id doesn't have an index yet.
        """
        return self.id_to_index[the_id]

    def get_id(self, the_index):
        """Get the id for the matrix/vector index"""
        return self.index_to_id[the_index]

    def size(self):
        """Size of the map"""
        return self.curr_index + 1

    def get_or_issue_index(self, the_id):
        """Get the index for the id or issue a new one if it doesn't exist"""
        if the_id in self.id_to_index:
            return self.id_to_index[the_id]
        else:
            self.curr_index += 1
            self.id_to_index[the_id] = self.curr_index
            self.index_to_id[self.curr_index] = the_id
            return self.curr_index

    def __repr__(self):
        return repr(self.index_to_id)


class UserItemMatrix(object):
    """A container class to represent a user to item preference matrix

    In users are assumed to be the rows and items the columns of the
    underlying matrix (mat)
    """

    def __init__(self, mat, user_bimap, item_bimap):
        self.mat = mat
        self.user_bimap = user_bimap
        self.item_bimap = item_bimap

    def build_item_similarity_matrix(self):
        """Generate a item-to-item similarity matrix, using cosine similarity"""
        item_mat = fast_cosine_similarity_matrix(self.mat.T)
        return ItemSimilarityMatrix(item_mat, self.item_bimap)

    def build_user_similarity_matrix(self):
        """Generate a user-to-user similarity matrix, using cosine similarity"""
        user_mat = fast_cosine_similarity_matrix(self.mat)
        return UserSimilarityMatrix(user_mat, self.user_bimap)

    def get_dense_pref_vector(self, user_id):
        """Get a N (number of items) x 1 numpy matrix of user preferences

        This method is equivalent to PreferenceVector#dense, but pulls the
        vector vector directly from the matrix, which allows the caller
        to discard the preference vectors once the matrix has been constructed.
        """
        return self.mat[self.user_bimap.get_index(user_id), :].T


class ItemSimilarityMatrix(object):
    """A container class for an item-to-item similarity matrix"""

    def __init__(self, mat, item_bimap):
        self.mat = mat
        self.item_bimap = item_bimap


class UserSimilarityMatrix(object):
    """A container class for an user-to-user similarity matrix"""

    def __init__(self, mat, user_bimap):
        self.mat = mat
        self.user_bimap = user_bimap


def build_user_item_matrix(pref_vectors, score_cb=None):
    """Build a user-item matrix from a list of preference vectors

    Users are rows, items are columns. This convention is used in all
    other functions in this package.

    Arguments:
      pref_vectors: list[PreferenceVector], a colelction of preference vectors,
        note that the data structure is iterated over twice so some collections
        won't work
      score_cb: numeric -> numeric, an optional callback to modify scores

    Returns:
      A UserItemMatrix
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
    return UserItemMatrix(np.matrix(mat), user_idxs, item_idxs)


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


# TODO: move to item recommender
def top_n_similar_items(item_id, item_sim_mat, item_idxs, n=10):
    item_idx = item_idxs.get_index(item_id)
    top_n_idxs = _top_n_plus_one(item_sim_mat[item_idx, :], n)
    return [item_idxs.get_id(idx) for idx in top_n_idxs if idx != item_idx]

# TODO: move to item recommender
def _top_n_plus_one(_1d_mat, n):
    return _1d_mat.A1.argsort()[-1:-(n + 2):-1].tolist()
