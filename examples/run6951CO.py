#!/usr/bin/python

"""
    Example to run pypot with an image from NGC 6951
"""

##== Properties for the galaxy
Distdef = 35.0        # Galaxy distance in Mpc
PAdef=138.7          # Position Angle
Incldef=41.5          # Inclination

"""
Import of the required modules
"""
import pypot
from pypot.pypot_main import *
import pylab as p
import numpy as num
import scipy
from scipy.ndimage.interpolation import rotate

dir_input = "/soft/pytorque/examples/data_6951/"

import pytorque
from pytorque.pytorque import discmodel

gal = discmodel(folder="/soft/python/pypot/examples/data_6951/", massimage="r6951nicmos_f160w.fits", gasimage="co21-un-2sigma-m0.fits", Vcfile="rot-co21un-01.tex", verbose=True, plot=True, distance=35.0, PA=138.7, inclination=41.5, cen=(178,198), cengas=(148,123), box=(90,90), boxgas=(90,90), pixelsize=0.025, pixelsizegas=0.1, azoom=5, Nbins=90)
gal.run_torque()
