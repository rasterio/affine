Affine
======

Matrices describing affine transformation of the plane.

.. image:: https://travis-ci.org/sgillies/affine.svg?branch=master
    :target: https://travis-ci.org/sgillies/affine

.. image:: https://coveralls.io/repos/sgillies/affine/badge.svg
    :target: https://coveralls.io/r/sgillies/affine

The Affine package is derived from Casey Duncan's Planar package. Please see
the copyright statement in `affine/__init__.py <affine/__init__.py>`__.

Usage
-----

The 3x3 augmented affine transformation matrix for transformations in two
dimensions is illustrated below.

.. ::

  | x' |   | a  b  c | | x |
  | y' | = | d  e  f | | y |
  | 1  |   | 0  0  1 | | 1 |

Matrices can be created by passing the values ``a, b, c, d, e, f`` to the
``affine.Affine`` constructor or by using its ``identity()``,
``translation()``, ``scale()``, ``shear()``, and ``rotation()`` class methods.

.. code-block:: pycon

  >>> from affine import Affine
  >>> Affine.identity()
  Affine(1.0, 0.0, 0.0,
         0.0, 1.0, 0.0)
  >>> Affine.translation(1.0, 5.0)
  Affine(1.0, 0.0, 1.0,
         0.0, 1.0, 5.0)
  >>> Affine.scale(2.0)
  Affine(2.0, 0.0, 0.0,
         0.0, 2.0, 0.0)
  >>> Affine.shear(45.0, 45.0)  # decimal degrees
  Affine(1.0, 0.9999999999999999, 0.0,
         0.9999999999999999, 1.0, 0.0)
  >>> Affine.rotation(45.0)     # decimal degrees
  Affine(0.7071067811865476, -0.7071067811865475, 0.0,
         0.7071067811865475, 0.7071067811865476, 0.0)

These matrices can be applied to ``(x, y)`` tuples to obtain transformed
coordinates ``(x', y')``.

.. code-block:: pycon

  >>> Affine.translation(1.0, 5.0) * (1.0, 1.0)
  (2.0, 6.0)
  >>> Affine.rotation(45.0) * (1.0, 1.0)
  (1.1102230246251565e-16, 1.414213562373095)

They may also be multiplied together to combine transformations.

.. code-block:: pycon

  >>> Affine.translation(1.0, 5.0) * Affine.rotation(45.0)
  Affine(0.7071067811865476, -0.7071067811865475, 1.0,
         0.7071067811865475, 0.7071067811865476, 5.0)

Usage with GIS data packages
----------------------------

Georeferenced raster datasets use affine transformations to map from image
coordinates to world coordinates. The ``affine.Affine.from_gdal()`` class
method helps convert `GDAL GeoTransform
<http://www.gdal.org/classGDALDataset.html#af9593cc241e7d140f5f3c4798a43a668>`__,
sequences of 6 numbers in which the first and fourth are the x and y offsets
and the second and sixth are the x and y pixel sizes.

Using a GDAL dataset transformation matrix, the world coordinates ``(x, y)``
corresponding to the top left corner of the pixel 100 rows down from the
origin can be easily computed.

.. code-block:: pycon

  >>> geotransform = (-237481.5, 425.0, 0.0, 237536.4, 0.0, -425.0)
  >>> fwd = Affine.from_gdal(*geotransform)
  >>> col, row = 0, 100
  >>> fwd * (col, row)
  (-237481.5, 195036.4)

The reverse transformation is obtained using the ``~`` operator.

.. code-block:: pycon

  >>> rev = ~fwd
  >>> rev * fwd * (col, row)
  (0.0, 99.99999999999999)

