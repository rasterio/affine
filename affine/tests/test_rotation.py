from math import sqrt

from affine import Affine


def test_rotation_angle():
    """A positive angle rotates a vector counter clockwise

    (1.0, 0.0):

        |
        |
        |
        |
        0---------*

    Affine.rotation(45.0) * (1.0, 0.0) == (0.707..., 0.707...)

        |
        |      *
        |
        |
        0----------
    """
    x, y = Affine.rotation(45.0) * (1.0, 0.0)
    assert round(x) == round(sqrt(2.0) / 2.0)
    assert round(y) == round(sqrt(2.0) / 2.0)
