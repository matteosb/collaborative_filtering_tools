"""Core classes used in cf_lib"""


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

    def __repr__(self):
        return 'PreferenceVector({0}, {1})'.format(self.user_id, self.prefs)


class IndexBiMap(object):
    """A bidirectional map used to issue and track matrix indecies.

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
