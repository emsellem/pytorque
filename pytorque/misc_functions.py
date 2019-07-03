#!/usr/bin/python

"""
A set of useful functions to derive the gravitational potential via pytorque
"""

__version__ = '1.1.0 (25-06, 2019)'

## Changes -- 
##   25/06/19- EE - v1.1.0: Python 3
##   13/04/07- EE - v1.0.1: Addition of stop_program()

"""
Import of the required modules
"""
# Astropy
import astropy
from astropy.io import fits as pyfits

# Other important packages
import pylab as p  # Standard Libraries
import numpy as np
import os 
import scipy 

##################################
###  Test to stop program
##################################
def stop_program():
    """Small function to ask for input and stop if needed
    """
    ok = input("Press S to Stop, and any other key to continue...\n")
    if ok in ["S","s"]  :
        return True
    return False

##################################
###  sigma clipping Function
##################################
def sig_clip(data, nsigma=5.):
    datamean = np.mean(data, axis=None)
    datasigma = np.std(data, axis=None)
    return data[np.where(data < datamean+ nsigma * datasigma)]

##################################
###  SECH Function
##################################
def sech(z):
    return 1. / np.cosh(z)

##################################
###  SECH2 Function
##################################
def sech2(z):
    return (1. / np.cosh(z))**2.

##################################
### Run Example 
##################################
def run_example():
    file_vobs='rot-7217cd3_both.txt'

##################################
### Reading the Circular Velocity
### Data from file
##################################
dic_comments = {"ROTCUR": "!", "ASCII": "#"}

def get_vcirc(filename, Vcfile_type="ROTCUR", stepR=1.0):
    #--- Reading data
    radius = Vc = eVc = rfine = Vcfine = 0.
    if not os.path.isfile(filename) :
        print('OPENING ERROR: File {0} not found'.format(filename))
        status = -1
    else :
        Vcdata = np.loadtxt(filename, 
                comments=dic_comments[Vcfile_type]).T
        if Vcfile_type == "ROTCUR":
            selV = (Vcdata[7] == 0) & (Vcdata[6] == 0)
            radius = Vcdata[0][selV]
            Vc = Vcdata[4][selV]
            eVc = Vcdata[5][selV]
        elif Vcfile_type == "ASCII":
            radius = Vcdata[0]
            Vc = Vcdata[1][selV]
            eVc = np.zeros_like(Vc)
        else:
            print("ERROR: Vc file type not recognised")
            status = -2

        #--- Spline interpolation
        rmax = np.max(radius, axis=None)
        rfine = np.arange(0., rmax, stepR)
        coeff_spline = scipy.interpolate.splrep(radius, Vc, k=1)
        Vcfine = scipy.interpolate.splev(rfine, coeff_spline)
        status = 0
        
    return status, radius, Vc, eVc, rfine, Vcfine

