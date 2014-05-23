Affine
======

Affine transformation matrices

The Affine package is derived from Casey Duncan's Planar package. Please see
the copyright statement in [affine/__init__.py](affine/__init__.py).

## Usage

    from affine import Affine
    
    # Georeferenced image data uses affine transformations to map from image
    # coordinates to world coordinates.  GDAL's geotransform is a sequence of
    # 6 numbers, the first and fourth being the x and y offsets and the second
    # and sixth being the x and y resolutions.
    transform = Affine.from_gdal(-237481.5, 425.0, 0.0, 237536.4, 0.0, -425.0)
    
    # Transform the image coordinates (100, 100) to world coordinates
    pixel, line = 100, 100
    print transform * (pixel, line)
    # Output:
    # (-194981.5, 195036.4)

