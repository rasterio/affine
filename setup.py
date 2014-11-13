#!/usr/bin/env python

from setuptools import setup

try:  # http://bugs.python.org/issue15881
    import multiprocessing
    assert multiprocessing
except ImportError:
    pass

# Parse the version from the affine module.
version = None
with open('affine/__init__.py', 'r') as fp:
    for line in fp:
        if "__version__" in line:
            exec(line.replace('_', ''))
            break
if version is None:
    raise ValueError("Could not determine version")

with open('README.rst') as fp:
    readme = fp.read()

setup(name='affine',
      version=version,
      description="Matrices describing affine transformation of the plane",
      long_description=readme,
      classifiers=[],
      keywords='affine transformation matrix',
      author='Sean Gillies',
      author_email='sean@mapbox.com',
      url='https://github.com/sgillies/affine',
      license='BSD',
      package_dir={'': '.'},
      packages=['affine'],
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      )
