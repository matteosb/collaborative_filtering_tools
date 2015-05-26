from ..core import matrix
import numpy as np


class ItemItemCFRecommender(object):
    """Implements item-item collaborative filtering"""\

    def __init__(self, pref_vectors):
        self.user_item_mat = matrix.build_user_item_matrix(pref_vectors)
        self.sim_mat = self.user_item_mat.build_item_similarity_matrix().mat
        self.item_bimap = self.user_item_mat.item_bimap

    def top_n_similar_items(self, item_id, n=10):
        item_index = self.item_bimap.get_index(item_id)
        top_n_plus_1_similars = matrix.top_n_indices_for_vector(
            self.sim_mat[item_index, :], n + 1)
            # filter out the item_id and slice the array
            # just in case there are n+1 prefect matches
        return [self.item_bimap.get_id(idx) for idx in top_n_plus_1_similars
                if idx != item_index][:n]

    def recommendations_for_user(self, user_id, n=10):
        dense_pref = self.user_item_mat.get_dense_pref_vector(user_id)
        return self.__recommendations(dense_pref, n)

    def recommendations_for_pref_vector(self, pref_vector, n=10):
        return self.__recommendations(pref_vector.dense(self.item_bimap), n)

    def __recommendations(self, dense_pref_vector, n):
        scored_items = self.sim_mat * dense_pref_vector
        # remove items the user has already rated from the scored_items
        filtered = np.where(dense_pref_vector != 0, 0, scored_items)
        return [self.item_bimap.get_id(idx) for idx in
                matrix.top_n_indices_for_vector(np.matrix(filtered), n)]
