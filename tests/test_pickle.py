"""
Validate that instances of `affine.Affine()` can be pickled and unpickled.
"""

from multiprocessing import Pool
import pickle

import affine


def test_pickle():
    a1 = affine.Affine(1, 2, 3, 4, 5, 6)
    assert pickle.loads(pickle.dumps(a1)) == a1
    # specify different g, h, i values than defaults
    a2 = affine.Affine(1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert pickle.loads(pickle.dumps(a2)) == a2


def _mp_proc(x):
    # A helper function - needed for test_with_multiprocessing()
    # Can't be defined inside the test because multiprocessing needs
    # everything to be in __main__
    assert isinstance(x, affine.Affine)
    return x


def test_with_multiprocessing():
    a1 = affine.Affine(1, 2, 3, 4, 5, 6)
    a2 = affine.Affine(6, 5, 4, 3, 2, 1)
    results = Pool(2).map(_mp_proc, [a1, a2])
    for expected, actual in zip([a1, a2], results):
        assert expected == actual
