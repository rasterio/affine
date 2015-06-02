from codecs import open as codecs_open
from setuptools import setup, find_packages


# Parse the version from the affine module.
with open('affine/__init__.py') as f:
    for line in f:
        if "__version__" in line:
            version = line.split("=")[1].strip()
            version = version.strip('"').strip("'")
            break

with codecs_open('README.rst', encoding='utf-8') as f:
    readme = f.read()


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
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      extras_require = {'test':  ['pytest']}
      )
