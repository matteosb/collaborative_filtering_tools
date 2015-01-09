import numpy as np

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
        # uses items to support deletion from the vector while iterating
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

    def has_id(self, the_id):
        return the_id in self.id_to_index

    def get_id(self, the_index):
        return self.index_to_id[the_index]

    def highest_index(self):
        return self.curr_index

    def size(self):
        return self.curr_index + 1

    def issue_index(self, the_id):
        if the_id not in self.id_to_index:
            self.curr_index += 1
            self.id_to_index[the_id] = self.curr_index
            self.index_to_id[self.curr_index] = the_id
            return self.curr_index
        else:
            raise IndexExistsError("Todo")

    def __str__(self):
        return str(self.index_to_id)


def build_user_item_mat_from_pref_vectors(pref_vectors):
    user_idxs = IndexBiMap()
    item_idxs = IndexBiMap()
    for vector in pref_vectors:
        user_idxs.get_or_issue_index(vector.user_id)
        for item, score in vector.iter_prefs():
            item_idxs.issue_index(item)
    # We loop through the data twice here but build a dense matrix
    # the alternatives are:
    # 1) loop once, build a sparse matrix
    #     and convert to dense (numpy likes dense matricies)
    # 2) if dimensions are known, build the zero matrix first,
    #     loop once and fill in
    mat = np.zeros((user_idxs.size(), item_idxs.size()), dtype=dtype)
    for vector in user_id_vector_map.values():
        row = user_idxs.get_index(vector.user_id)
        for item, score in vector.iter_prefs():
            column = item_idxs.get_index(item)
            mat[row][column] = score_cb(score)
    return np.matrix(mat), user_idxs, item_idxs



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
    norms = np.sqrt(np.diag(sim_mat))
    # Compute cosines by dividing by norms (sim matrix is assumed to be M^TM or MM^T, so the diagonal is the norm^2)
    # there might be a better way to divide by the norms using fancy numpy slicing/broadcast rules
    # matrix is modified in place
    for i in xrange(len(norms)):
        sim_mat[i, :] = sim_mat[i, :]/norms[i]
        sim_mat[:, i] = sim_mat[:, i]/norms[i]
    return sim_mat

class IndexExistsError(Exception):
    pass
