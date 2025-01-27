import math

import pytest

from affine import Affine


def test_rotation_angle():
    """A positive angle rotates a vector counter clockwise

    (1.0, 0.0):

        |
        |
        |
        |
        0---------*

    Affine.rotation(45.0) @ (1.0, 0.0) == (0.707..., 0.707...)

        |
        |      *
        |
        |
        0----------
    """
    x, y = Affine.rotation(45.0) @ (1.0, 0.0)
    sqrt2div2 = math.sqrt(2.0) / 2.0
    assert x == pytest.approx(sqrt2div2)
    assert y == pytest.approx(sqrt2div2)


def test_rotation_matrix():
    """A rotation matrix has expected elements

    | cos(a) -sin(a) |
    | sin(a)  cos(a) |

    """
    deg = 90.0
    rot = Affine.rotation(deg)
    rad = math.radians(deg)
    cosrad = math.cos(rad)
    sinrad = math.sin(rad)
    assert rot.a == pytest.approx(cosrad)
    assert rot.b == pytest.approx(-sinrad)
    assert rot.c == 0.0
    assert rot.d == pytest.approx(sinrad)
    assert rot.e == pytest.approx(cosrad)
    assert rot.f == 0.0


def test_rotation_matrix_pivot():
    """A rotation matrix with pivot has expected elements"""
    rot = Affine.rotation(90.0, pivot=(1.0, 1.0))
    exp = (
        Affine.translation(1.0, 1.0)
        @ Affine.rotation(90.0)
        @ Affine.translation(-1.0, -1.0)
    )
    for r, e in zip(rot, exp):
        assert r == pytest.approx(e)
