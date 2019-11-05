#!/usr/bin/env python
"""pytorque: gravitational potential of galaxies

pytorque is small set of modules aimed at computing the gravitational potential
and force torques in the equatorial plane given an input FITS images of a galaxy. 
It uses a simple Fourier transform derivation for the potential, with the
optional fit of a bulge component before the deprojection occurs.
"""
from __future__ import absolute_import, division, print_function

import sys

# Version
version = {}
with open("pytorque/version.py") as fp:
    exec(fp.read(), version)

# simple hack to allow use of "python setup.py develop".  Should not affect
# users, only developers.
if 'develop' in sys.argv:
    # use setuptools for develop, but nothing else
    from setuptools import setup
else:
    from distutils.core import setup

import os

if os.path.exists('MANIFEST'): 
    os.remove('MANIFEST')

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(name='pytorque',
      description='Python Gravitational Potential Module',
      version = version['__version__'],
      author='Eric Emsellem',
      author_email='eric.emsellem@eso.org',
      maintainer='Eric Emsellem',
      url='http://',
      packages=['pytorque'],
#      package_dir={'pytorque'},
     )
