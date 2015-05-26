"""Core object models for cf_lib"""
import numpy as np


class PreferenceVector(object):
    """General representation of a user-item preference vector

    PreferenceVectors form the backbone of cf_lib. A collection of
    preference vectors specific to your domain is the main entry point
    into various collaborative filtering recommenders.

    Example of "preferences" are user ratings or purchase history.
    """

    def __init__(self, user_id, pref_dict=None):
        self.user_id = user_id
        self.prefs = pref_dict or {}

    def add(self, item_id, score):
        """Add an item preference score"""
        self.prefs[item_id] = score

    def get(self, item_id):
        """Get an item preference score"""
        return self.prefs[item_id]

    def remove(self, item_id):
        """Remove an item preference score"""
        del self.prefs[item_id]

    def iter_prefs(self):
        """Iterate over preference scores"""
        # uses items() to support deletion from the vector while iterating
        for item_id, score in self.prefs.items():
            yield item_id, score

    def dense(self, item_bimap):
        """Create a dense #items x 1 numpy matrix for the

        Arguments:
          item_bimap: IndexBiMap for the items, this is used
            to determine the overall size and index of the preferences
            in the resulting numpy matrix.
        """
        arr = np.zeros(item_bimap.size())
        for item_id, score in self.iter_prefs():
            index = item_bimap.get_index(item_id)
            arr[index] = score
        return np.matrix(arr).T

    def __repr__(self):
        return 'PreferenceVector({0}, {1})'.format(self.user_id, self.prefs)
