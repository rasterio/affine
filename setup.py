#!/usr/bin/env python

from setuptools import setup

# Parse the version from the affine module.
with open('affine/__init__.py') as fp:
    for line in fp:
        if "__version__" in line:
            version = line.split("=")[1].strip()
            version = version.strip('"').strip("'")
            break

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
      # test_suite='nose.collector',  # use pytest
      )
