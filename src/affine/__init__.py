"""Affine transformation matrices.

The Affine package is derived from Casey Duncan's Planar package. See the
copyright statement below.
"""

#############################################################################
# Copyright (c) 2010 by Casey Duncan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name(s) of the copyright holders nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#############################################################################

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from functools import cached_property
import math
import warnings

from attrs import astuple, define, field

__all__ = ["Affine"]
__author__ = "Sean Gillies"
__version__ = "3.0rc3"

EPSILON: float = 1e-5
EPSILON2: float = 1e-10


class AffineError(Exception):
    pass


class TransformNotInvertibleError(AffineError):
    """The transform could not be inverted."""


class UndefinedRotationError(AffineError):
    """The rotation angle could not be computed for this transform."""


def cos_sin_deg(deg: float) -> tuple[float, float]:
    """Return the cosine and sin for the given angle in degrees.

    With special-case handling of multiples of 90 for perfect right
    angles.
    """
    deg = deg % 360.0
    if deg == 90.0:
        return 0.0, 1.0
    if deg == 180.0:
        return -1.0, 0
    if deg == 270.0:
        return 0, -1.0
    rad = math.radians(deg)
    return math.cos(rad), math.sin(rad)


@define(frozen=True)
class Affine:
    """Two dimensional affine transform for 2D linear mapping.

    Parameters
    ----------
    a, b, c, d, e, f, [g, h, i] : float
        Coefficients of the 3 x 3 augmented affine transformation
        matrix.

    Attributes
    ----------
    a, b, c, d, e, f, g, h, i : float
        Coefficients of the 3 x 3 augmented affine transformation
        matrix::

            | x' |   | a  b  c | | x |
            | y' | = | d  e  f | | y |
            | 1  |   | g  h  i | | 1 |

        `g`, `h`, and `i` are always 0, 0, and 1.

    The Affine package is derived from Casey Duncan's Planar package.
    See the copyright statement below.  Parallel lines are preserved by
    these transforms. Affine transforms can perform any combination of
    translations, scales/flips, shears, and rotations.  Class methods
    are provided to conveniently compose transforms from these
    operations.

    Internally the transform is stored as a 3x3 transformation matrix.
    The transform may be constructed directly by specifying the first
    two rows of matrix values as 6 floats. Since the matrix is an affine
    transform, the last row is always ``(0, 0, 1)``.

    N.B.: multiplication of a transform and an (x, y) vector *always*
    returns the column vector that is the matrix multiplication product
    of the transform and (x, y) as a column vector, no matter which is
    on the left or right side. This is obviously not the case for
    matrices and vectors in general, but provides a convenience for
    users of this class.

    """

    a: float = field(converter=float)
    b: float = field(converter=float)
    c: float = field(converter=float)
    d: float = field(converter=float)
    e: float = field(converter=float)
    f: float = field(converter=float)

    # The class has 3 attributes that don't have to be specified: g, h,
    # and i. If they are, the given value has to be the same as the
    # default value. This allows a new instances to be created from the
    # tuple form of another, like Affine(*Affine.identity()).

    g: float = field(default=0.0, converter=float)

    @g.validator
    def _check_g(self, attribute, value):
        if value != 0.0:
            raise ValueError("g must be equal to 0.0")

    h: float = field(default=0.0, converter=float)

    @h.validator
    def _check_h(self, attribute, value):
        if value != 0.0:
            raise ValueError("h must be equal to 0.0")

    i: float = field(default=1.0, converter=float)

    @i.validator
    def _check_i(self, attribute, value):
        if value != 1.0:
            raise ValueError("i must be equal to 1.0")

    @classmethod
    def from_gdal(
        cls, c: float, a: float, b: float, f: float, d: float, e: float
    ) -> Affine:
        """Use same coefficient order as GDAL's GetGeoTransform().

        Parameters
        ----------
        c, a, b, f, d, e : float
            Parameters ordered by GDAL's GeoTransform.

        Returns
        -------
        Affine
        """
        return cls(a, b, c, d, e, f)

    @classmethod
    def identity(cls) -> Affine:
        """Return the identity transform.

        Returns
        -------
        Affine
        """
        return identity

    @classmethod
    def translation(cls, xoff: float, yoff: float) -> Affine:
        """Create a translation transform from an offset vector.

        Parameters
        ----------
        xoff, yoff : float
            Translation offsets in x and y directions.

        Returns
        -------
        Affine
        """
        return cls(1.0, 0.0, xoff, 0.0, 1.0, yoff)

    @classmethod
    def scale(cls, *scaling: float) -> Affine:
        """Create a scaling transform from a scalar or vector.

        Parameters
        ----------
        *scaling : float or sequence of two floats
            One or two scaling factors. A scalar value will scale in both
            dimensions equally. A vector scaling value scales the dimensions
            independently.

        Returns
        -------
        Affine
        """
        if len(scaling) == 1:
            sx = scaling[0]
            sy = sx
        else:
            sx, sy = scaling
        return cls(sx, 0.0, 0.0, 0.0, sy, 0.0)

    @classmethod
    def shear(cls, x_angle: float = 0.0, y_angle: float = 0.0) -> Affine:
        """Create a shear transform along one or both axes.

        Parameters
        ----------
        x_angle, y_angle : float
            Shear angles in degrees parallel to the x- and y-axis.

        Returns
        -------
        Affine
        """
        mx = math.tan(math.radians(x_angle))
        my = math.tan(math.radians(y_angle))
        return cls(1.0, mx, 0.0, my, 1.0, 0.0)

    @classmethod
    def rotation(cls, angle: float, pivot: Sequence[float] | None = None) -> Affine:
        """Create a rotation transform at the specified angle.

        Parameters
        ----------
        angle : float
            Rotation angle in degrees, counter-clockwise about the pivot point.
        pivot : sequence of float (px, py), optional
            Pivot point coordinates to rotate around. If None (default), the
            pivot point is the coordinate system origin (0.0, 0.0).

        Returns
        -------
        Affine
        """
        ca, sa = cos_sin_deg(angle)
        if pivot is None:
            return cls(ca, -sa, 0.0, sa, ca, 0.0)
        px, py = pivot
        # fmt: off
        return cls(
            ca, -sa, px - px * ca + py * sa,
            sa, ca, py - px * sa - py * ca,
        )
        # fmt: on

    @classmethod
    def permutation(cls, *scaling: float) -> Affine:
        """Create the permutation transform.

        For 2x2 matrices, there is only one permutation matrix that is
        not the identity.

        Parameters
        ----------
        *scaling : any
            Ignored.

        Returns
        -------
        Affine
        """
        return cls(0.0, 1.0, 0.0, 1.0, 0.0, 0.0)

    def __array__(self, dtype=None, copy: bool | None = None):
        """Get affine matrix as a 3x3 NumPy array.

        Parameters
        ----------
        dtype : data-type, optional
            The desired data-type for the array.
        copy : bool, optional
            If None (default) or True, a copy of the array is always returned.
            If False, a ValueError is raised as this is not supported.

        Returns
        -------
        array

        Raises
        ------
        ValueError
            If ``copy=False`` is specified.
        """
        import numpy as np

        if copy is False:
            raise ValueError("`copy=False` isn't supported. A copy is always created.")
        return np.array(self._astuple, dtype=(dtype or float)).reshape((3, 3))

    def __str__(self) -> str:
        """Concise string representation."""
        return (
            f"|{self.a: .2f},{self.b: .2f},{self.c: .2f}|\n"
            f"|{self.d: .2f},{self.e: .2f},{self.f: .2f}|\n"
            f"|{self.g: .2f},{self.h: .2f},{self.i: .2f}|"
        )

    def __repr__(self) -> str:
        """Precise string representation."""
        return (
            f"Affine({self.a!r}, {self.b!r}, {self.c!r},\n"
            f"       {self.d!r}, {self.e!r}, {self.f!r})"
        )

    def to_gdal(self) -> tuple[float, float, float, float, float, float]:
        """Return same coefficient order expected by GDAL's SetGeoTransform().

        Returns
        -------
        tuple
            Ordered: c, a, b, f, d, e.
        """
        return (self.c, self.a, self.b, self.f, self.d, self.e)

    def to_shapely(self) -> tuple[float, float, float, float, float, float]:
        """Return affine transformation parameters for shapely's affinity module.

        Returns
        -------
        tuple
            Ordered: a, b, d, e, c, f.
        """
        return (self.a, self.b, self.d, self.e, self.c, self.f)

    @property
    def xoff(self) -> float:
        """Alias for 'c'."""
        return self.c

    @property
    def yoff(self) -> float:
        """Alias for 'f'."""
        return self.f

    @cached_property
    def determinant(self) -> float:
        """Evaluate the determinant of the transform matrix.

        This value is equal to the area scaling factor when the
        transform is applied to a shape.

        Returns
        -------
        float
        """
        return self.a * self.e - self.b * self.d

    @property
    def _scaling(self) -> tuple[float, float]:
        """The absolute scaling factors of the transformation.

        This tuple represents the absolute value of the scaling factors of the
        transformation, sorted from bigger to smaller.
        """
        a, b, d, e = self.a, self.b, self.d, self.e

        # The singular values are the square root of the eigenvalues
        # of the matrix times its transpose, M M*
        # Computing trace and determinant of M M*
        trace = a**2 + b**2 + d**2 + e**2
        det2 = (a * e - b * d) ** 2

        delta = trace**2 / 4.0 - det2
        if delta < EPSILON2:
            delta = 0.0

        sqrt_delta = math.sqrt(delta)
        l1 = math.sqrt(trace / 2.0 + sqrt_delta)
        l2 = math.sqrt(trace / 2.0 - sqrt_delta)
        return l1, l2

    @property
    def eccentricity(self) -> float:
        """The eccentricity of the affine transformation.

        This value represents the eccentricity of an ellipse under
        this affine transformation.

        Raises
        ------
        NotImplementedError
            For improper transformations.
        """
        l1, l2 = self._scaling
        return math.sqrt(l1**2 - l2**2) / l1

    @property
    def rotation_angle(self) -> float:
        """The rotation angle in degrees of the affine transformation.

        This is the rotation angle in degrees of the affine transformation,
        assuming it is in the form M = R S, where R is a rotation and S is a
        scaling.

        Raises
        ------
        UndefinedRotationError
            For improper and degenerate transformations.
        """
        if self.is_proper or self.is_degenerate:
            l1, _ = self._scaling
            y, x = self.d / l1, self.a / l1
            return math.degrees(math.atan2(y, x))
        raise UndefinedRotationError

    @property
    def is_identity(self) -> bool:
        """True if this transform equals the identity matrix, within rounding limits."""
        return self is identity or self.almost_equals(identity, EPSILON)

    @property
    def is_rectilinear(self) -> bool:
        """True if the transform is rectilinear.

        i.e., whether a shape would remain axis-aligned, within rounding
        limits, after applying the transform.
        """
        return (abs(self.a) < EPSILON and abs(self.e) < EPSILON) or (
            abs(self.d) < EPSILON and abs(self.b) < EPSILON
        )

    @property
    def is_conformal(self) -> bool:
        """True if the transform is conformal.

        i.e., if angles between points are preserved after applying the
        transform, within rounding limits.  This implies that the
        transform has no effective shear.
        """
        return abs(self.a * self.b + self.d * self.e) < EPSILON

    @property
    def is_orthonormal(self) -> bool:
        """True if the transform is orthonormal.

        Which means that the transform represents a rigid motion, which
        has no effective scaling or shear. Mathematically, this means
        that the axis vectors of the transform matrix are perpendicular
        and unit-length.  Applying an orthonormal transform to a shape
        always results in a congruent shape.
        """
        a, b, d, e = self.a, self.b, self.d, self.e
        return (
            self.is_conformal
            and abs(1.0 - (a * a + d * d)) < EPSILON
            and abs(1.0 - (b * b + e * e)) < EPSILON
        )

    @cached_property
    def is_degenerate(self) -> bool:
        """Return True if this transform is degenerate.

        A degenerate transform will collapse a shape to an effective area
        of zero, and cannot be inverted.

        Returns
        -------
        bool
        """
        return self.determinant == 0.0

    @cached_property
    def is_proper(self) -> bool:
        """Return True if this transform is proper.

        A proper transform (with a positive determinant) does not include
        reflection.

        Returns
        -------
        bool
        """
        return self.determinant > 0.0

    @property
    def column_vectors(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """The values of the transform as three 2D column vectors.

        Returns
        -------
        tuple of three tuple pairs
            Ordered (a, d), (b, e), (c, f).
        """
        return (self.a, self.d), (self.b, self.e), (self.c, self.f)

    def almost_equals(self, other: Affine, precision: float | None = None) -> bool:
        """Compare transforms for approximate equality.

        Parameters
        ----------
        other : Affine
            Transform being compared.
        precision : float, default EPSILON
            Precision to use to evaluate equality.

        Returns
        -------
        bool
            True if absolute difference between each element
            of each respective transform matrix < ``precision``.
        """
        precision = precision or EPSILON
        return all(abs(sv - ov) < precision for sv, ov in zip(self, other))

    @cached_property
    def _astuple(self) -> tuple[float]:
        return astuple(self)

    def __getitem__(self, index):
        return self._astuple[index]

    def __iter__(self):
        return iter(self._astuple)

    def __len__(self):
        return 9

    def __gt__(self, other) -> bool:
        return NotImplemented

    __ge__ = __lt__ = __le__ = __gt__

    # Override from base class. We do not support entrywise
    # addition, subtraction or scalar multiplication because
    # the result is not an affine transform

    def __add__(self, other):
        raise TypeError("Operation not supported")

    __iadd__ = __add__

    def __matmul__(self, other):
        """Matrix multiplication.

        Apply the transform using matrix multiplication, creating
        a resulting object of the same type.  A transform may be applied
        to another transform, a vector, vector array, or shape.

        Parameters
        ----------
        other : Affine or iterable of (vx, vy, [vw])

        Returns
        -------
        Affine or a tuple of two or three items
        """
        sa, sb, sc, sd, se, sf = self[:6]
        if isinstance(other, Affine):
            oa, ob, oc, od, oe, of = other[:6]
            return self.__class__(
                sa * oa + sb * od,
                sa * ob + sb * oe,
                sa * oc + sb * of + sc,
                sd * oa + se * od,
                sd * ob + se * oe,
                sd * oc + se * of + sf,
            )
        # vector of 2 or 3 items
        try:
            num_items = len(other)
        except (TypeError, ValueError):
            return NotImplemented
        if num_items == 2:
            vx, vy = other
        elif num_items == 3:
            vx, vy, vw = other
            vw_eq_one = vw == 1.0
            try:
                is_eq_one = bool(vw_eq_one)
                msg = "third value must be 1.0"
            except ValueError:
                is_eq_one = (vw_eq_one).all()
                msg = "third values must all be 1.0"
            if not is_eq_one:
                raise ValueError(msg)
        else:
            raise TypeError("expected vector of 2 or 3 items")
        px = vx * sa + vy * sb + sc
        py = vx * sd + vy * se + sf
        if num_items == 2:
            return (px, py)
        return (px, py, vw)

    def __rmatmul__(self, other):
        return NotImplemented

    def __imatmul__(self, other):
        if not isinstance(other, Affine):
            raise TypeError("Operation not supported")
        return NotImplemented

    def __mul__(self, other):
        """Multiplication.

        Apply the transform using matrix multiplication, creating
        a resulting object of the same type.  A transform may be applied
        to another transform, a vector, vector array, or shape.

        Parameters
        ----------
        other : Affine or iterable of (vx, vy)

        Returns
        -------
        Affine or a tuple of two items
        """
        # TODO: consider enabling this for 3.1
        # warnings.warn(
        #     "Use `@` matmul instead of `*` mul operator for matrix multiplication",
        #     PendingDeprecationWarning,
        #     stacklevel=2,
        # )
        if isinstance(other, Affine):
            return self.__matmul__(other)
        try:
            _, _ = other
            return self.__matmul__(other)
        except (ValueError, TypeError):
            return NotImplemented

    def __rmul__(self, other):
        return NotImplemented

    def __imul__(self, other):
        if isinstance(other, tuple):
            warnings.warn(
                "in-place multiplication with tuple is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )
        return NotImplemented

    def itransform(self, seq: MutableSequence[Sequence[float]]) -> None:
        """Transform a sequence of points or vectors in-place.

        Parameters
        ----------
        seq : mutable sequence

        Returns
        -------
        None
            The input sequence is mutated in-place.
        """
        if self is not identity and self != identity:
            sa, sb, sc, sd, se, sf = self[:6]
            for i, (x, y) in enumerate(seq):
                seq[i] = (x * sa + y * sb + sc, x * sd + y * se + sf)

    def __invert__(self):
        """Return the inverse transform.

        Raises
        ------
        TransformNotInvertible
            If the transform is degenerate.
        """
        if self.is_degenerate:
            raise TransformNotInvertibleError("Cannot invert degenerate transform")
        idet = 1.0 / self.determinant
        sa, sb, sc, sd, se, sf = self[:6]
        ra = se * idet
        rb = -sb * idet
        rd = -sd * idet
        re = sa * idet
        # fmt: off
        return self.__class__(
            ra, rb, -sc * ra - sf * rb,
            rd, re, -sc * rd - sf * re,
        )
        # fmt: on

    def __getnewargs__(self):
        """Pickle protocol support.

        Notes
        -----
        Normal unpickling creates a situation where __new__ receives all
        9 elements rather than the 6 that are required for the
        constructor.  This method ensures that only the 6 are provided.
        """
        return self[:6]


identity = Affine(1, 0, 0, 0, 1, 0)
"""The identity transform"""

# Miscellaneous utilities


def loadsw(s: str) -> Affine:
    """Return Affine from the contents of a world file string.

    This method also translates the coefficients from center- to
    corner-based coordinates.

    Parameters
    ----------
    s : str
        String with 6 floats ordered in a world file.

    Returns
    -------
    Affine
    """
    if not hasattr(s, "split"):
        raise TypeError("Cannot split input string")
    coeffs = s.split()
    if len(coeffs) != 6:
        raise ValueError(f"Expected 6 coefficients, found {len(coeffs)}")
    a, d, b, e, c, f = (float(x) for x in coeffs)
    center = Affine(a, b, c, d, e, f)
    return center @ Affine.translation(-0.5, -0.5)


def dumpsw(obj: Affine) -> str:
    """Return string for a world file.

    This method also translates the coefficients from corner- to
    center-based coordinates.

    Returns
    -------
    str
    """
    center = obj @ Affine.translation(0.5, 0.5)
    return "\n".join(repr(getattr(center, x)) for x in list("adbecf")) + "\n"


def set_epsilon(epsilon: float) -> None:
    """Set the global absolute error value and rounding limit.

    This value is accessible via the affine.EPSILON global variable.

    Parameters
    ----------
    epsilon : float
        The global absolute error value and rounding limit for
        approximate floating point comparison operations.

    Returns
    -------
    None

    Notes
    -----
    The default value of ``0.00001`` is suitable for values that are in
    the "countable range". You may need a larger epsilon when using
    large absolute values, and a smaller value for very small values
    close to zero. Otherwise approximate comparison operations will not
    behave as expected.
    """
    global EPSILON, EPSILON2
    EPSILON = float(epsilon)
    EPSILON2 = EPSILON**2


set_epsilon(1e-5)
