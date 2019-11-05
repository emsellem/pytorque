"""Copyright (C) 2007 Max Planck Institute fuer Astronomie (MPIA)
                & Centre de Recherche Astronomique de Lyon (CRAL)

print pytorque.__LICENSE__  for the terms of use

This set of modules in intended to provide the gravitational
potential and torques of a galaxy, using a fits images as the input
for the luminosity distribution.

Submodules:
=========
misc_functions:
   Series of useful functions including sech profiles

pytorque:
   Main class, discmodel, used to derive the potential

"""
__LICENSE__ = """
Copyright (C) 2007  Max Planck Institute fuer Astronomie (MPIA)
              & Centre de Recherche Astronomique de Lyon (CRAL)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    3. The name of AURA and its representatives may not be used to
      endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY MPIA/CRAL ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

"""
from .version import __date__, __version__

import sys, os

"""
 Trying to load the different needed modules
"""
try : 
   import pylab
except ImportError: 
    print('There is a problem with importing pylab at initialisation')
try : 
    import astropy
except ImportError: 
    print('There is a problem with importing astropy at initialisation')

"""
Import the different submodules
"""
from pytorque import (torque, misc_functions)
