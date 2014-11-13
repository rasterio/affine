from setuptools import setup, find_packages
import sys, os

# Parse the version from the fiona module.
with open('affine/__init__.py') as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

readme = open('README.rst').read()

setup(name='affine',
      version=version,
      description="Matrices describing affine transformation of the plane",
      long_description=readme,
      classifiers=[],
      keywords='',
      author='Sean Gillies',
      author_email='sean@mapbox.com',
      url='https://github.com/sgillies/affine',
      license='BSD',
      package_dir={'': '.'},
      packages=['affine'],
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          # -*- Extra requirements: -*-
      ],
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
