"""Test interoperability with NumPy."""

import pytest

from affine import Affine

try:
    import numpy as np
    from numpy import testing
except ImportError:
    pytest.skip("requires numpy", allow_module_level=True)


def test_array():
    tfm = Affine(*np.linspace(0.1, 0.6, 6))
    tfm_ar = np.array(tfm)
    expected = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.0, 0.0, 1.0],
        ],
    )
    assert tfm_ar.shape == (3, 3)
    assert tfm_ar.dtype == np.float64
    testing.assert_allclose(tfm_ar, expected)

    # dtype option
    tfm_ar = np.array(tfm, dtype=np.float32)
    assert tfm_ar.shape == (3, 3)
    assert tfm_ar.dtype == np.float32
    testing.assert_allclose(tfm_ar, expected)

    # copy option
    tfm_ar = np.array(tfm, copy=True)  # default None does the same

    # Behaviour of copy=False is different between NumPy 1.x and 2.x
    if int(np.version.short_version.split(".", 1)[0]) >= 2:
        with pytest.raises(ValueError, match="A copy is always created"):
            np.array(tfm, copy=False)
    else:
        testing.assert_allclose(np.array(tfm, copy=False), expected)
