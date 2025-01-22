"""Test interoperability with NumPy."""

import pytest

from affine import Affine

try:
    import numpy as np
    from numpy import testing
except ImportError:
    pytest.skip("requires numpy", allow_module_level=True)


def test_array():
    a, b, c, d, e, f = (np.arange(6) + 1) / 10
    tfm = Affine(a, b, c, d, e, f)
    expected = np.array(
        [
            [a, b, c],
            [d, e, f],
            [0, 0, 1],
        ],
    )
    ar = np.array(tfm)
    assert ar.shape == (3, 3)
    assert ar.dtype == np.float64
    testing.assert_allclose(ar, expected)

    # dtype option
    ar = np.array(tfm, dtype=np.float32)
    assert ar.shape == (3, 3)
    assert ar.dtype == np.float32
    testing.assert_allclose(ar, expected)

    # copy option
    ar = np.array(tfm, copy=True)  # default None does the same
    testing.assert_allclose(ar, expected)

    # Behaviour of copy=False is different between NumPy 1.x and 2.x
    if int(np.version.short_version.split(".", 1)[0]) >= 2:
        with pytest.raises(ValueError, match="A copy is always created"):
            np.array(tfm, copy=False)
    else:
        testing.assert_allclose(np.array(tfm, copy=False), expected)

    # cross-check some properties
    ar = np.array(tfm)
    assert tfm.determinant == pytest.approx(np.linalg.det(ar))
    np.testing.assert_allclose(~tfm, np.linalg.inv(ar))

    # check 3x3 array, with unexpected g, h, i values
    a, b, c, d, e, f, g, h, i = [7, 2, 1, 0, 3, -1, -3, 4, -2]
    tfm = Affine(a, b, c, d, e, f, g, h, i)
    ar = np.array(tfm)
    assert ar.shape == (3, 3)
    assert ar.dtype == np.float64
    testing.assert_array_equal(
        ar,
        [
            [a, b, c],
            [d, e, f],
            [g, h, i],
        ],
    )

    # only numpy can calculate 3x3 answers
    assert np.linalg.det(ar) == pytest.approx(1.0)
    np.testing.assert_allclose(
        [
            [-2.0, 8.0, -5.0],
            [3.0, -11.0, 7.0],
            [9.0, -34.0, 21.0],
        ],
        np.linalg.inv(ar),
    )

    # Affine's properties assume 2x2 arrays
    # thus have different answers than numpy
    ar2x2 = np.array(
        [
            [a, b, c],
            [d, e, f],
            [0, 0, 1],
        ],
    )
    assert tfm.determinant == pytest.approx(np.linalg.det(ar2x2))
    np.testing.assert_allclose(~tfm, np.linalg.inv(ar2x2))
