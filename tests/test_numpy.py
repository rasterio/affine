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
    testing.assert_array_equal(ar, expected)

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


def test_linalg():
    # cross-check properties with numpy's linear algebra module
    ar = np.array(
        [
            [0, -2, 2],
            [3, 0, 5],
            [0, 0, 1],
        ]
    )
    tfm = Affine(*ar.flatten())
    assert tfm.determinant == pytest.approx(6.0)
    assert np.linalg.det(ar) == pytest.approx(6.0)

    expected_inv = np.array(
        [
            [0, 1 / 3, -5 / 3],
            [-1 / 2, 0, 1],
            [0, 0, 1],
        ]
    )
    testing.assert_allclose(~tfm, expected_inv)
    testing.assert_allclose(np.linalg.inv(ar), expected_inv)


def test_matmul_3x3_array():
    A = Affine(2, 0, 3, 0, 3, 2)
    Ar = np.array(A)

    # matrix @ matrix = matrix
    res = A @ Affine.identity()
    assert isinstance(res, Affine)
    testing.assert_equal(res, Ar)
    res = Ar @ np.eye(3)
    assert isinstance(res, np.ndarray)
    testing.assert_equal(res, Ar)


def test_matmul_vector():
    A = Affine(2, 0, 3, 0, 3, 2)
    Ar = np.array(A)

    # matrix @ vector = vector
    v = (2, 3, 1)
    vr = np.array(v)
    expected_p = (7, 11, 1)
    res = A @ v
    assert isinstance(res, tuple)
    testing.assert_equal(res, expected_p)
    res = A @ vr
    assert isinstance(res, tuple)
    testing.assert_equal(res, expected_p)
    res = Ar @ vr
    assert isinstance(res, np.ndarray)
    testing.assert_equal(res, expected_p)


def test_matmul_items():
    A = Affine(2, 0, 3, 0, 3, 2) @ Affine.rotation(45)
    shape = (4, 5)
    vx, vy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    expected_px = np.array(
        [
            [3.0, 4.4142137, 5.8284273, 7.2426405, 8.656855],
            [1.5857865, 3.0, 4.4142137, 5.8284273, 7.2426405],
            [0.17157288, 1.5857865, 3.0, 4.4142137, 5.8284273],
            [-1.2426407, 0.17157288, 1.5857865, 3.0, 4.4142137],
        ]
    )
    expected_py = np.array(
        [
            [2.0, 4.1213202, 6.2426405, 8.363961, 10.485281],
            [4.1213202, 6.2426405, 8.363961, 10.485281, 12.606602],
            [6.2426405, 8.363961, 10.485281, 12.606602, 14.727922],
            [8.363961, 10.485281, 12.606602, 14.727922, 16.849243],
        ]
    )
    px, py = A @ (vx, vy)
    testing.assert_allclose(px, expected_px)
    testing.assert_allclose(py, expected_py)

    px, py, pw = A @ (vx, vy, 1)
    testing.assert_allclose(px, expected_px)
    testing.assert_allclose(py, expected_py)
    assert pw == 1

    px, py, pw = A @ (vx, vy, np.ones(shape))
    testing.assert_allclose(px, expected_px)
    testing.assert_allclose(py, expected_py)
    testing.assert_array_equal(pw, np.ones(shape))

    # see GH-125 with mul `*` op
    px, py = A * (vx, vy)
    testing.assert_allclose(px, expected_px)
    testing.assert_allclose(py, expected_py)


def test_matmul_item_errors():
    shape = (4, 5)
    vx, vy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    with pytest.raises(ValueError, match="third value must be 1.0"):
        Affine.identity() @ (vx, vy, 2)
    with pytest.raises(ValueError, match="third values must all be 1.0"):
        Affine.identity() @ (vx, vy, np.zeros(shape))
    with pytest.raises(TypeError, match="2 or 3 items"):
        Affine.identity() @ (vx, vy, np.ones(shape), np.ones(shape))


def test_mul_item_errors():
    shape = (4, 5)
    vx, vy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    with pytest.raises(TypeError):
        Affine.identity() * (vx, vy, 1)
    with pytest.raises(TypeError):
        Affine.identity() * (vx, vy, np.zeros(shape))
