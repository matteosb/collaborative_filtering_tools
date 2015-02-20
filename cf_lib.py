import numpy


class PreferenceVector(object):
    """General representation of a item preference vector."""
    def __init__(self, user_id, pref_dict=None):
        self.user_id = user_id
        self.prefs = pref_dict or {}

    def add(self, item_id, score):
        self.prefs[item_id] = score

    def get(self, item_id):
        self.prefs[item_id]

    def remove(self, item_id):
        del self.prefs[item_id]

    def iter_prefs(self):
        # uses items() to support deletion from the vector while iterating
        for item_id, score in self.prefs.items():
            yield item_id, score

    def __repr__(self):
        return "PreferenceVector({0}, {1})".format(self.user_id, self.prefs)


class IndexBiMap(object):
    """A bidirectional map used to issue and track matrix indecies.

    Assists creating dense matrecies by converting arbitrary ids to indecies
    and indecies back to ids.
    """

    def __init__(self):
        self.id_to_index = {}
        self.index_to_id = {}
        self.curr_index = -1

    def get_index(self, the_id):
        return self.id_to_index[the_id]

    def get_id(self, the_index):
        return self.index_to_id[the_index]

    def size(self):
        return self.curr_index + 1

    def get_or_issue_index(self, the_id):
        if the_id in self.id_to_index:
            return self.id_to_index[the_id]
        else:
            self.curr_index += 1
            self.id_to_index[the_id] = self.curr_index
            self.index_to_id[self.curr_index] = the_id
            return self.curr_index

    def __repr__(self):
        return repr(self.index_to_id)


def build_matrix_from_pref_vectors(user_vectors, score_cb=None):
    if score_cb is None:
        score_cb = lambda x: x
    user_idxs = IndexBiMap()
    item_idxs = IndexBiMap()
    for vector in user_vectors:
        user_idxs.get_or_issue_index(vector.user_id)
        for item, _ in vector.iter_prefs():
            item_idxs.get_or_issue_index(item)
    # We loop through the data twice here but build a dense matrix
    # the alternatives are:
    # 1) loop once, build a sparse matrix
    #     and convert to dense (numpy likes dense matricies)
    # 2) if dimensions are known, build the zero matrix first,
    #     loop once and fill in
    mat = numpy.zeros((user_idxs.size(), item_idxs.size()), dtype=numpy.float32)
    for vector in user_vectors:
        row = user_idxs.get_index(vector.user_id)
        for item, score in vector.iter_prefs():
            column = item_idxs.get_index(item)
            mat[row][column] = score_cb(score)
    return numpy.matrix(mat), user_idxs, item_idxs


def compute_item_to_item_cos_sim_mat(mat):
    """Generate a item-to-item similarity matrix, using cosine similarity"""
    sims = mat.T * mat
    _normalize_sim_mat(sims)
    return sims


def compute_user_to_user_cos_sim_mat(mat):
    """Generate a user-to-user similarity matrix, using cosine similarity"""
    sims = mat * mat.T
    _normalize_sim_mat(sims)
    return sims


def _normalize_sim_mat(sim_mat):
    norms = numpy.sqrt(numpy.diag(sim_mat))
    denominators = numpy.outer(norms, norms)
    # divides in place
    numpy.divide(sim_mat, denominators, out=sim_mat)
    return sim_mat


def _top_n_plus_one(_1d_mat, n):
    return _1d_mat.A1.argsort()[-1:-(n + 2):-1].tolist()


def top_n_similar_items(item_id, item_sim_mat, item_idxs, n=10):
    item_idx = item_idxs.get_index(item_id)
    top_n_idxs = _top_n_plus_one(item_sim_mat[item_idx, :], n)
    return [item_idxs.get_id(idx) for idx in top_n_idxs if idx != item_idx]


def get_user_vector_from_user_item_mat(id_, user_item_mat, user_idxs):
    return user_item_mat[user_idxs.get_index(id_), :].T
