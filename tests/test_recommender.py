from cf_tools.core import PreferenceVector, matrix
from cf_tools.recommender import ItemItemCFRecommender
import numpy as np
import random

NUM_USERS = 1000
NUM_ITEMS = 500
POSSIBLE_RATINGS = range(1, 6)

def random_rating():
    # withold some ratings randomly
    return random.choice(POSSIBLE_RATINGS) * random.randint(0, 1)

def build_vectors(nu=NUM_USERS, ni=NUM_ITEMS):
    for u in xrange(nu):
        p = PreferenceVector(u)
        for i in xrange(ni):
            r = random_rating()
            if r: p.add(i, r)
        yield p

def test_foo():
    print "***********"
    vectors = list(build_vectors())
    print ItemItemCFRecommender(vectors).recommendations_for_user(5)
    assert 1 == 0
