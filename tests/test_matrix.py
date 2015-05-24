from cf_tools.core import matrix, PreferenceVector
import numpy as np

from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_equal, assert_almost_equal


USERS = ['u1', 'u2', 'u3']
ITEMS = ['i1', 'i2']
PREF_VECTORS = [
    PreferenceVector('u1', {'i1': .5}),
    PreferenceVector('u2', {'i2': 1}),
    PreferenceVector('u3', {'i1': .5, 'i2': .5}),
]
UI_MATRIX = matrix.build_user_item_matrix(PREF_VECTORS)


def verify_item_bimap(matrix_model):
    assert_equal(set(matrix_model.item_bimap.id_to_index.keys()), set(ITEMS))


def verify_user_bimap(matrix_model):
    assert_equal(set(matrix_model.user_bimap.id_to_index.keys()), set(USERS))


def verify_matrix_is_symmetric(mat):
    assert_array_equal(mat, mat.T)


def test_build_user_item_matrix():
    verify_user_bimap(UI_MATRIX)
    verify_item_bimap(UI_MATRIX)
    # this assertion is a little brittle since it assumes
    # a certain (non-essential) order of processing the preference vectors
    expected_mat = np.matrix('.5 0; 0 1; .5 .5')
    assert_array_equal(UI_MATRIX.mat, expected_mat)


def test_build_item_similarity_matrix():
    imat = UI_MATRIX.build_item_similarity_matrix()
    verify_item_bimap(imat)
    assert_equal(imat.mat.shape, (2, 2))
    assert_array_almost_equal(np.diag(imat.mat), np.array([1, 1]))
    verify_matrix_is_symmetric(imat.mat)
    expected_cosine = np.sqrt(10) / 10
    assert_almost_equal(imat.mat[0, 1], expected_cosine)


def test_build_user_similarity_matrix():
    umat = UI_MATRIX.build_user_similarity_matrix()
    verify_user_bimap(umat)
    assert_equal(umat.mat.shape, (3, 3))
    assert_array_almost_equal(np.diag(umat.mat), np.array([1, 1, 1]))
    verify_matrix_is_symmetric(umat.mat)
    cosine_1_2 = 0
    cosine_1_3 = np.sqrt(2) / 2
    cosine_2_3 = np.sqrt(2) / 2
    assert_almost_equal(umat.mat[0, 1], cosine_1_2)
    assert_almost_equal(umat.mat[0, 2], cosine_1_3)
    assert_almost_equal(umat.mat[1, 2], cosine_2_3)
