#!/usr/bin/python

"""
Program for computing the gravitational potential from a combination 
of optical and near-infrared images. 
MPIA / CRAL: @ 2007
Authors: Sebastian Haan
         Eric Emsellem
"""

__version__ = '0.9.5 (25-06, 2019)'

## Changes -- 
##   25/06/19- EE - v0.9.5: Cleaning for Python3
##   14/04/07- EE - v0.9.4: Debugging. Correction of deprojection
##                                   Change of rotate function
##                                   Simplification of some routines
##   13/04/07- EE - v0.9.3: cleaning, redefinition of Width in X,Y
##                                   Simplification of some routines
"""
Import of the required modules
"""

# general modules
import sys
import os 
from os.path import join as joinpath

# Astropy and fits
import astropy
from astropy.io import fits as pyfits
from astropy.constants import G as Ggrav # m^3 *kg^-1 * s^-2

# std packages
import pylab as p  # Standard Libraries
import numpy as np
from numpy import float32

import scipy 
from scipy.fftpack.basic import fft, fft2, fftn, rfft, ifft, ifft2, ifftn, irfft
from scipy.signal.signaltools import convolve, convolve2d, fftconvolve 
from scipy.ndimage.morphology import morphological_gradient
from scipy.ndimage.interpolation import zoom
from scipy import interpolate
from scipy.ndimage.interpolation import rotate  
from scipy.ndimage.interpolation import affine_transform

## Importing all function from the pytorque module
from .misc_functions import *

##############################################################
#----- Extracting the header and data array ------------------
##############################################################
def extract_data(image_name, pixelsize=1., verbose=True):
    """Extract 2D data array from fits
    and return the data and the header

    Parameters
    ----------
    image_name: string
                Name of fits image
    verbose:    boolean
                Default is True

    Returns
    -------
    data:       numpy array
                data array from the input image
    h:          header
                Fits header from the input image
    """
    if os.path.isfile(image_name):
        if verbose :
            print(("Opening the Input image: {0}".format(image_name)))
        #--------Reading of fits-file for grav. pot----------
        data = pyfits.getdata(image_name)
        h = pyfits.getheader(image_name)
        #-------------- Fits Header IR Image------------
        naxis1, naxis2 = h['NAXIS1'], h['NAXIS2']
        data = np.nan_to_num(data.reshape((naxis2, naxis1)))

        ##== Checking the step from the Input image (supposed to be in degrees)
        ##== If it doesn't exist, we set the step to 1. (arcsec)
        desc = 'cdelt1'
        if desc in h:
            steparc = np.fabs(h[desc] * 3600.) # calculation in arcsec
            if verbose :
                print('Read pixel size ({0}) of Main Image = {1}'.format(h[desc], steparc))
        else :
            steparc = pixelsize # in arcsec
            if verbose :
                print("Didn't find a CDELT descriptor, use step={0}".format(steparc))
        return data, h, steparc
    
    else :
        print(('Filename {0} does not exists, sorry!'.format(image_name)))
        return None, None, 1.0
#============================================================

##############################################################
#-----Rotation and Deprojecting routine----------------------
##############################################################
def deproject_frame(data, PA, inclination=90.0, plot=False):
    """Returns a deprojected frame given a PA and inclination

    Parameters
    ----------
    data:
    PA:
    inclination:
    plot: 

    Returns
    -------

    """
    
    # Reading the shape of the disk array
    Ysize, Xsize = data.shape

    # Creating the new set of needed arrays
    disk_dpj=np.zeros((Ysize+1,Xsize+1))
    disk_rot=np.zeros_like(disk_dpj)
    disk_rec=np.zeros_like(disk_dpj)
    disk_dpj_c=np.zeros((Ysize, Xsize))
    
    # Recentering the disk
    disk_rec[:Ysize,:Xsize] = data[:,:]
    if plot:
        print("Image to deproject has shape: %f, %f" %(Ysize, Xsize))
        p.figure(1)
        p.clf()
        p.imshow(disk_rec)
        p.title("Transfer of the Image before deprojection")
        p.draw()
        if stop_program() : sys.exit(0)
    
    # Deprojection Matrix
    phi= np.deg2rad(inclination)
    dpj_matrix = np.array([[1.0 * np.cos(phi),   0.   ],
                          [0.0, 1.0 ]])   
 
    # Rotate Disk around theta
    disk_rot =  rotate(np.asarray(disk_rec), PA - 90., reshape=False)
 
    # Deproject Image
    offy = Ysize / 2 - 1. - (Ysize / 2 - 1.) * np.cos(phi)
    disk_dpj_c = affine_transform(disk_rot, dpj_matrix, 
                                  offset=(offy,0))[:Ysize,:Xsize]
 
    return disk_dpj_c

#============================================================
#-----------Create Radial Profile----------------------------
#============================================================
def comp_profile(r, Data, Nbins, verbose=True): 
    """ Input:
                  input Data array 
                  array with radial coordinate
                  and number of radial bins
        Output: 
                  radial array (1D)
                  Mean intensity array (1D)
    """
    if verbose :
        print("Deriving the radial profile ... \n")
    ##== First deriving the max and cutting it in Nbins
    maxr = np.max(r, axis=None)/2
    stepr = maxr / (Nbins-1)
    rsamp = np.arange(0., maxr+stepr, stepr)
    y = np.zeros_like(rsamp)
    ##== Filling in the values for y (only if there are some selected pixels)
    for i in range(len(rsamp)-1):
        sel = np.where((r >= rsamp[i]) & (r < rsamp[i+1]))  ##== selecting an annulus between two bins
        if len(sel) > 0 :
            y[i] = np.mean(Data[sel], axis=None)

    ##-- Returning the obtained profile
    return rsamp, y
##############################################################
######### GALAXYMODEL CLASS ##################################
##############################################################
class discmodel(object)   :
    """Galaxy model class
    Attributes
    ----------
    folder : str
        file folder where the input images are
    massimage : str
        Name of Mass image (fits)
    gasimage :  str
        Name of gas distribution image (fits)
    Vcfile : str
        rotation file
    distance : float
        Distance of galaxy in Mpc [1.]
    inclination : float
        Inclination of galaxy in degrees [0.]
    plot: bool
        Whether to plot or not [True]
    verbose : bool
        Verbose option [False]
    """
    def __init__(self, folder="", massimage=None, gasimage=None, 
                 Vcfile=None, Vcfile_type="ROTCUR",
                 verbose=False, distance=1., PA=0., inclination=0., 
                 cen=(512,512), cengas=(512,512), 
                 box=(300,300), boxgas=(300,300), 
                 pixelsize=1., pixelsizegas=1., azoom=6, 
                 plot=True, Nbins=100):
        """Initialisation of the disc model class
        Mainly setting up a few parameters like the distance, PA and inclination
        and reading the data from the given input images

        Parameters
        ----------
        PA: float
            Position Angle from North, in degrees

        distance: float
            Distance in Mpc

        """

        ##=== Checking if the directory and file name exist.
        ##=== For this we use the os python module
        self.verbose = verbose
        self.plot = plot

        ##=== Some useful number
        self.distance = distance # Galaxy distance in Mpc
        self.pc_per_arcsec = self.distance * np.pi / 0.648  # Conversion arcsec => pc (pc per arcsec)
        self.PA = PA                                        # Position angle in degrees
        self.inclination = inclination                      # Inclination in degrees

        ##== Checking the existence of the two images ===========================================
        if not os.path.exists(folder):
            print(('ERROR: Path %s' %(folder), ' does not exists, sorry!'))
            return
        self.folder = folder 

        if massimage is None :
            print('ERROR: no filename for the main Input Image provided')
            return
        if gasimage is None :
            print('ERROR: no filename for the gas Input Image provided')
            return
        if Vcfile is None :
            print('ERROR: no filename for the Rotation Input File provided')
            return

        ##== We now save these information into the class
        if self.verbose:
            print("Trying to open Input files now...")
        self.massimage = massimage
        inimage_name = joinpath(self.folder, self.massimage)
        self.gasimage = gasimage
        gasimage_name = joinpath(self.folder, self.gasimage)
        self.Vcfile = Vcfile
        self.Vcfile_type = Vcfile_type
        self.file_Vcobs = joinpath(self.folder, self.Vcfile)
        
        # Extracting the data from the 2 images
        self.data, h, self.steparc = extract_data(inimage_name, pixelsize, self.verbose)
        self.data_gas, h, self.steparc2 = extract_data(gasimage_name, pixelsizegas, self.verbose)

        self.zoomfactor = self.steparc2 / self.steparc
        if self.verbose:
            print(("zoomfactor (pixel size ratio) = {0}".format(self.zoomfactor)))

        #---------------Change some galaxy properties here--------------------
        self.steppc = self.steparc * self.pc_per_arcsec   # Pixel size in pc
        self.steppc2 = self.steparc2 * self.pc_per_arcsec # Pixel size of gas image in pc
        self.cen = cen                                        # Position for the centre
        self.box = box 
        self.cengas = cengas                                        # Position for the centre of gas image
        self.boxgas = boxgas                                        # Box coordinates
        self.Nbins = Nbins

        ##== Checking if the zoom is larger than 1 before proceeding
        if azoom > 1 :
            self.azoom = azoom
            if self.verbose:
                print("Zoom will be {0:d}".format(self.azoom)) 
        else :
            print("ERROR: zoom factor (azoom) should be larger than 1!")
            return

        self.init_disks()
        if self.read_Vc() < 0:
            print("ERROR: reading Vc file did not succeed")
    
    def run_torques(self, softening=0):
        """Run the series of processes needed to calculate
        the torques 
        """
        self.sub_background()
        self.deproject_disk()
        self.radial_excl(self.rpc, self.diskS, self.Nbins)

        # Bulge fitting
        self.bulge_fit(self.diskS)
        ## Substraction of bulge from projected disk 
        self.bulge_sub(self.diskN)

        ## Deprojection of disk without bulge
        self.deproject_disk_nobulge()
        ## Adding bulge component again to deprojected disk
        self.bulge_add(self.diskS_nobulge)

        ##Radial Profile
        self.rdiskpc, self.prof_disk = comp_profile(self.rpc, self.diskF, self.Nbins)

        ## Deriving the kernel
        self.calc_kernel(softening=softening)
        ##Calculation of potential
        self.calc_pot()
        ##Calculation of forces (gradient of potential) - unscaled
        self.calc_force()
        ##Create rotation velocity field, unscaled here - from force field
        self.VcU = self.calc_vrot_field()

        ##Create rotation velocity profile, unscaled here
        self.rsampVc, self.sampVcU = comp_profile(self.rpc, self.VcU, Nbins=self.Nbins)
        ## Calculation of radial dependent M to L ratio by comparison with observed velocity
        ## mode 1: radial dependent, else constant scaling value
        self.scale_Vc(self.VcU, mode=0)

        ##-- We can now scale the potential and forces
        self.potC = self.ML * self.pot
        self.FxC = self.ML * self.Fx
        self.FyC = self.ML * self.Fy
        self.FradC = self.ML * self.Frad
        self.FtanC = self.ML * self.Ftan
        self.VcC = np.sqrt(self.ML) * self.VcU

        #Decomposition of potential
        self.decompose(self.potC, Nbins=45, fc_order=8)

        ##Calculation of torques
        self.calc_torque(self.rpc, self.VcC, self.FxC, self.FyC, self.Nbins)
        
        ## plotting the result
        if self.plot:
            p.clf()
            p.plot(self.torque_r,self.torque_mean)  # Create plot of radialprofile
            p.xlabel("R")
            p.ylabel("Torque")
            p.draw()
            ok = input("Type any key to continue")

            p.clf()
            vmax = np.median(np.ravel(self.torque)) + np.std(self.torque, axis=None) * 5
            vmin = np.median(np.ravel(self.torque)) - np.std(self.torque, axis=None) * 5
            im = p.imshow(rotate(self.torque, 90. - self.PA), vmin=vmin, vmax=vmax)
            p.colorbar()
            p.title("Radial Torques")
            p.draw()
        
    def init_disks(self):
        """Initialise the arrays for the disks
        """
        Xwidth = self.box[0] 
        Ywidth = self.box[1]
        [Ywidth_z, Xwidth_z] = [Ywidth * self.azoom, 
                                Xwidth * self.azoom] ##== Zoomed box size
        ##== Total size will be twice this
        self.Ytotdisk, self.Xtotdisk = Ywidth_z*2, Xwidth_z*2
        ##== And defining the new center
        self.xcen_z, self.ycen_z = Xwidth_z, Ywidth_z   
        ##== Allocating the right space
        self.disk = np.zeros((self.Xtotdisk, self.Ytotdisk))  
        self.gasdisk = np.zeros_like(self.disk)

        if self.verbose:
            print("New Center = {0:d}, {1:d} - "
                  "Box size = {2:d}, {3:d}".format(
                      self.xcen_z, self.ycen_z, 
                      self.Xtotdisk, self.Ytotdisk))

        self.Rdiskpx = self.box[0] / 2.0        #Should be changed later eventually

        ##== Getting the data within the defined box
        ##== Note that array has length =  2 times Xwidth (resp. Ywidth) and the center at Xwidth (Ywidth)
        ##== We are therefore only filling the inner part of the new array with the original data

        ## Useful (non zero) original part of the data
        self.usediskX = [self.xcen_z - Xwidth, self.xcen_z + Xwidth]
        self.usediskY = [self.ycen_z - Ywidth, self.ycen_z + Ywidth]

        self.disk[self.usediskX[0]:self.usediskX[1],
                  self.usediskY[0]:self.usediskY[1]] = \
                          self.data[self.cen[0] - Xwidth: self.cen[0] + Xwidth,
                                    self.cen[1] - Ywidth: self.cen[1] + Ywidth] 

        if self.plot: 
            p.figure(1)
            p.clf()
            p.imshow(self.data)
            p.title("Original Imaging data")
            p.figure(2)
            p.imshow(self.disk)
            p.title("Transfered data after zooming/box")
            p.draw()
            if stop_program() : sys.exit(0)
            p.close(2)
        
        ## Converting cellsize of GAS array to same cell-size as Input Main Image array#
        self.data_gas_z = zoom(self.data_gas, self.zoomfactor)
        if self.plot:
            p.figure(1)
            p.clf()
            p.imshow(self.data_gas)
            p.title("Original gas data")
            p.figure(2)
            p.imshow(self.data_gas_z)
            p.title("Zoomed Gas data")
            p.draw()
            if stop_program() : sys.exit(0)
            p.close(2)
        xcen_zoom = np.round(self.cengas[0] * self.zoomfactor).astype(np.int)
        ycen_zoom = np.round(self.cengas[1] * self.zoomfactor).astype(np.int)
        Xwidth_gas_z = np.round(self.boxgas[0] * self.zoomfactor).astype(np.int)
        Ywidth_gas_z = np.round(self.boxgas[1] * self.zoomfactor).astype(np.int)

        ##== Now taking the right width to extract the gas info
        if self.verbose:
            print("Rescaled Gas data had +/- {0}, {1} pixels".format(Xwidth_gas_z*2, Ywidth_gas_z*2))
            print("To be included in +/- {0}, {1} pixels".format(Xwidth_z*2, Ywidth_z*2))
        minXwidth = np.minimum(Xwidth_gas_z, Xwidth_z)
        minYwidth = np.minimum(Ywidth_gas_z, Ywidth_z)

        ## Specifying the array for gas distribution
        if self.verbose:
            print("Pixel Coordinates for the gas data")
            print(self.ycen_z-minYwidth, self.ycen_z+minYwidth, 
                    self.xcen_z-minXwidth, self.xcen_z+minXwidth)
            print("Corresponding pixels in new Gas array")
            print(ycen_zoom-minYwidth, ycen_zoom+minYwidth, 
                    xcen_zoom-minXwidth, xcen_zoom+minXwidth)
        self.gasdisk[self.ycen_z-minYwidth:self.ycen_z+minYwidth, 
                     self.xcen_z-minXwidth:self.xcen_z+minXwidth] = \
                             self.data_gas_z[ycen_zoom-minYwidth:ycen_zoom+minYwidth, 
                                             xcen_zoom-minXwidth:xcen_zoom+minXwidth]
        
        ## Simple Deprojecting of gas disk (including rotation)
        self.gasdisk_dep = deproject_frame(self.gasdisk, self.PA, self.inclination)
        if self.plot :
            p.figure(1)
            p.clf()
            p.imshow(self.gasdisk)
            p.title("Zoomed gas data")
            p.figure(2)
            p.imshow(self.gasdisk_dep)
            p.title("Deprojected Gas data")
            p.draw()
            if stop_program() : sys.exit(0)
            p.close(2)
                
    #============================================================
    #-----Calculation and  Subtraction of Backround--------------
    #============================================================
    def sub_background(self) :
        """ Subtract the background by simply removing the average on one corner
        """
        self.background = np.mean(self.disk[self.usediskY[0]:self.usediskY[0]+20,
                                            self.usediskX[0]:self.usediskX[0]+20], 
                                            axis=None) 
        if self.verbose :
             print("Background value found to be {0}".format(self.background))
             print("Subtracting this background value... \n")

        self.diskN = self.disk - self.background

        ##-- Replacing all values below 0. with 0.
        self.diskN[np.where(self.diskN < 0.)] = 0.

    #============================================================
    #--Deprojecting of the disk (includes rotation of the major axis to the north axis)--
    #============================================================
    def deproject_disk(self) :
        """ Deproject the data using the normalised disk intensity, the PA and
             inclination
        """
        if self.verbose :
            print("Deprojecting the disk ... \n")
        self.diskS = deproject_frame(self.diskN, self.PA, self.inclination)
        if self.plot :
            p.figure(1)
            p.clf()
            p.imshow(self.diskS)
            p.title("Deprojected disk")
            p.figure(2)
            p.imshow(self.diskN)
            p.title("Original disk (bulge subtracted)")
            p.draw()
            if stop_program() : sys.exit(0)
            p.close(2)

        ##== Calculate Cartesian coordinates x,y in pc in respect to the center (coord:0,0):
        ##== No central pixel
        xpix, ypix = self.diskS.shape
        xran = np.arange(- xpix / 2.+ 0.5, xpix / 2.+ 0.5, 1.0)
        yran = np.arange(- ypix / 2.+ 0.5, ypix / 2.+ 0.5, 1.0)
        self.xpc, self.ypc = np.meshgrid(xran * self.steppc,yran * self.steppc)
        
        self.rpc = np.sqrt(self.xpc**2 + self.ypc**2)        # cylindrical cordinates: radius r 
        self.theta = np.zeros_like(self.rpc)                # and theta( counting from east axis anticlockwise)
        indtheta = np.where(self.ypc>0)             
        self.theta[indtheta] = np.arccos(self.xpc[indtheta] / self.rpc[indtheta])
        indtheta = np.where(self.ypc<0)
        self.theta[indtheta] = 2.0 * np.pi - np.arccos(self.xpc[indtheta] / self.rpc[indtheta]) 


        
    #============================================================
    #-----------Create Radial Profile excluding a theta region --
    #============================================================
    def radial_excl(self, r, Data, Nbins):   
        """ Input:
                      input Data array 
                      array with radial coordinate
                      and number of radial bins
                      and theta cut around major axis
             Output: 
                      radial array (1D)
                      Mean intensity array (1D)
        """
        if self.verbose :
            print("Deriving the radial profile for bulge... \n")
        ##== First deriving the max and cutting it in Nbins
        maxr = np.max(r, axis=None) / 2
        stepr = maxr / (Nbins - 1)
        rsamp = np.arange(0., maxr + stepr, stepr)
        y = np.zeros_like(rsamp)
        ##== Filling in the values for y (only if there are some selected pixels)
        for i in range(len(rsamp)-1):
            ##== selecting an annulus between two bins
            sel = np.where((r >= rsamp[i]) & (r < rsamp[i+1]) 
                    & (((self.theta > np.pi * 1. / 6.) 
                       & (self.theta < np.pi * 5. / 6.)) 
                    | ((self.theta > np.pi * 7. / 6.) 
                       & (self.theta < np.pi * 11. / 6.))))  
            if len(sel) > 0 :
                y[i] = np.mean(Data[sel], axis=None)

        ##== Saving this into the discmodel class
        self.rbin_excl = rsamp
        self.rmean_excl = y
        if self.plot:
            p.clf()
            p.plot(self.rbin_excl, np.log10(self.rmean_excl + 1.e-2))
            p.xlabel("R")
            p.ylabel("Radial Profile (in Log10, with Theta cut)")
            p.draw()
            if stop_program() : sys.exit(0)

    #============================================================
    #-----------Bulge component treatment------
    #============================================================
    def bulge_fit(self, disk):
            
        if self.verbose :
            print("Fitting the disk and the bulge... \n")
        Xwidth, Ywidth = self.box[0]*self.steppc, self.box[1]*self.steppc
        x = self.rbin_excl
        ind = np.where(x < Xwidth)
        x = x[ind]
        y_meas = self.rmean_excl[ind]
        I_max = np.max(disk, axis=None)
        Imax_bulge, Rbulge, Imax_disk, Rdisk = I_max*0.5, 0.02*Xwidth, I_max*0.5, 0.4*Xwidth  #Initial values
        #I_bulge = Imax_bulge * pow((1+(x/Rbulge)**2),(-5/2)) #Rgo Plummer Sphere
        I_bulge = Imax_bulge * 4/3 * Rbulge**5 * pow((x**2 + Rbulge**2),(-2)) #projected Plummer Sphere
        I_disk  = Imax_disk * np.exp(-x/Rdisk)  #Model for Disk
        I_disk  = Imax_disk * Rdisk / (2*np.pi) * pow((x**2 + Rdisk**2),(-3/2)) #Kuzmin disk
        y_true = I_bulge + I_disk     #sum of both
                  
        def residuals(par, y, x):           # create residual equation: y_true - y_meas
            Imax_bulge, Rbulge, Imax_disk, Rdisk = par
            #err = y - (Imax_bulge * pow((1+(x/Rbulge)**2),(-5/2)) + Imax_disk * np.exp(-x/Rdisk))
            err = y - ( Imax_bulge * Rbulge**5 * 4/3 * pow((x**2 + Rbulge**2),(-2)) + Imax_disk * np.exp(-x/Rdisk) )
            return err
                  
        def peval_all(x, par):
            #return par[0] * pow((1+(x/par[1])**2),(-5/2)) + par[2] * np.exp(-x/par[3])
            return par[0] * par[1]**5 * 4/3 * pow((x**2 + (par[1])**2),(-2)) + par[2] * np.exp(-x/par[3])
        
        def peval_bulge(x, par):
            #return par[0] * pow((1+(x/par[1])**2),(-5/2))
            return par[0] *  par[1]**5 * 4/3 * pow((x**2 + (par[1])**2),(-2))
        
        def peval_disk(x, par):
            #return par[2] * np.exp(-x/par[3])
            return par[2] * np.exp(-x/par[3])
                  
        par0 = [I_max*0.8, 0.05*Xwidth, I_max*0.2, 0.5*Xwidth]  #Initial conditions
        if self.verbose:
            print("Initial conditions for bulge fit : ", par0)

        #from scipy.optimize.tnc import fmin_tnc
        from scipy.optimize import leastsq
        self.plsq = leastsq(residuals,par0, args=(y_meas, x),maxfev=10000)
        
        self.bulge = peval_bulge(self.rpc,self.plsq[0])
                                        
        if self.plot:
            p.clf()
            line = p.plot(x,np.log10(y_meas+1.e-2), 'k',x,np.log10(peval_bulge(x,self.plsq[0])+1.e-2),'b', x,np.log10(peval_disk(x,self.plsq[0])+1.e-2), 'r',) #,r_bulge,bulge_mean)
            p.legend(line, ("Total", "Bulge", "Disk"))
            p.xlabel("R [pc]")
            p.ylabel("Flux, Bulge, Disk (log10)")
            p.draw()
            if stop_program() : sys.exit(0)
          
    def bulge_sub(self, disk):
        if self.verbose :
            print("Subtracting the bulge component... \n")
        self.disk_nobulge = disk - self.bulge
          
    def bulge_add(self, disk):
        if self.verbose :
            print("Adding the bulge to the deprojected disk ... \n")
        self.diskF = disk + self.bulge

    def deproject_disk_nobulge(self) :
        if self.verbose :
            print("Deprojecting the disk without bulge... \n")
        self.diskS_nobulge = deproject_frame(self.disk_nobulge, self.PA, self.inclination)
          
    #============================================================
    #----------------Calculation of the Kernel Function----------
    #============================================================
    def calc_kernel(self, softening=0., function="sech2"):                       
        """ Input: radial coordinates, maximal radius of disk
        """
        if self.verbose :
            print("Deriving the kernel ... \n")
        self.softening = softening                      # Softening Parameter, here set to 0
        self.hzpx = int(self.Rdiskpx / 12.)         # In grid size, the vertical height is taken as 1/12 of the disk radius

        ##== Grid in z from -hz to +hz with no central point
        self.zpx = np.arange(0.5 - self.hzpx, self.hzpx + 0.5, 1.) 
        self.zpc = self.zpx * self.steppc                # Z grid in pc

        if function == "sech" :                               # sech or sech2 function (1/cosh = sech)                                  
            self.h = sech(self.zpx / self.hzpx)           # Vertical thickening function in z, has to be normalized later 
        elif function == "sech2" :
            self.h = sech2(self.zpx / self.hzpx)          
        else :
            self.h = sech2(self.zpx / self.hzpx)          

        Sumh = np.sum(self.h, axis=None)              # Integral of h over the entire range
        self.hn = self.h / Sumh                               # Normalised function h

        kernel = self.hn[np.newaxis,np.newaxis,...] / (np.sqrt(self.rpc[...,np.newaxis]**2 + self.softening**2 + self.zpc**2)) 
        self.kernel = np.sum(kernel, axis=2)
        

    #============================================================
    #----------------Calculation of the potential----------
    #============================================================
    def calc_pot(self):
        #-----------Convolution with mass density by Fourier Transformation-----------------------------
        #Pot = -Ggrav.value * Mass_density *conv* kernel
        
        #Some other modules, but didn't worked so far
        #from scipy.ndimage.filters import convolve #equal to "_correlate_or_convolve
        #from scipy.fftpack.convolve import convolve, convolve_z, init_convolution_kernel  #fortran objects
        #from numpy import convolve  #only 1-D sequences
        if self.verbose :
                  print("Calculating the potential ... \n")
        r = self.rpc
        self.convol = np.zeros((len(r),len(r)))         # Array initialization for grav- potential
        #self.convol = fftconvolve(self.kernel, cmass_r, mode='same')
        self.convol = fftconvolve(self.kernel, self.diskF, mode='same') # Convolution of the kernel with the weight (mass density)
        ##== Saving this into the discmodel class
        self.pot = -Ggrav.value * self.convol 

    #============================================================
    #-------- Calculation of Circular Velocity field ------------
    #============================================================
    def calc_vrot_field(self):
        if self.verbose :
            print("Calculating the rotation velocities from force field... \n")
        return np.sqrt(np.fabs(self.rpc * self.Frad))

    def read_Vc(self, Vcfile=None, Vcfile_type="ROTCUR"):
        """Reading the input Vc file
        """
        if Vcfile is None:
            Vcfile = self.Vcfile
        if self.verbose :
            print("Reading the Vc file")
        ##--- Reading of observed rot velocities
        Vc_filename = joinpath(self.folder + Vcfile)
        status, self.Vcobs_r, self.Vcobs, self.eVcobs, \
                self.Vcobs_rint, self.Vcobs_int = get_vcirc(Vc_filename, 
                        Vcfile_type=Vcfile_type)

        if status == 0 * self.verbose:
            print("Vc file successfully read")
        return status

    #==============================================================
    #-Comparison with observed Rotation Curve and M/L normalization
    #==============================================================
    def scale_Vc(self, Vfield, mode=0):
        """Comparing with given rotation curve
        """
              
        if self.verbose :
            print("Comparison with observed rotation velocities... \n")

        ##--- Building a similar profile for the model
        Vmodel_unscaled = np.zeros_like(self.Vcobs_rint)
        Vfactor_profile = np.zeros_like(self.Vcobs_rint)
        VFactor = np.ones_like(self.rpc)

        ##--- Change radial scale from arcsec to pc scale
        Vcobs_rpc = self.Vcobs_r * self.pc_per_arcsec
        Vcobs_rintpc = self.Vcobs_rint * self.pc_per_arcsec
        maxrpc_obs = np.max(Vcobs_rintpc, axis=None)
        maxVcobs = np.max(self.Vcobs, axis=None)

        ##--- Rebinning to bin size of calculated velocities / 4
        ##--- Calculation of ratio of observed vel to calculated vel 

        ## If mode == 1 we proceed with the calculation of a radial M/L
        if mode==1 :
            for i in range(len(Vcobs_rintpc)-1):
                sel = np.where((self.rpc >= Vcobs_rintpc[i]) & (self.rpc < Vcobs_rintpc[i+1])) 
                if len(sel) > 0 :
                    ##-- Average of the Vfield within the annulus
                    Vmodel_unscaled[i] = np.mean(Vfield[sel], axis=None)
                    ##-- Ratio of the average observed Vc and the model one for that annulus
                    Vfactor_profile[i] =  (self.Vcobs_int[i] + self.Vcobs_int[i+1]) \
                                          / (2. * Vmodel_unscaled[i])
                    ##-- Saving the value in the Factor profile
                    VFactor[sel] = Vfactor_profile[i]

            ##--- Outside the range of observed circular velocity we keep the
            ##--- average found in the last 1/10th of the range
            VFactor9_10 = VFactor[np.where(Vcobs_rintpc[i] > (9. * maxrpc_obs / 10.))]
            VFactor_out = np.mean(VFactor9_10, axis=None)
            VFactor[np.where(self.rpc >= maxrpc_obs)] = VFactor_out

        ## If mode != 1 we assume the ratio of the peak values for M/L
        else :
            ##--- Constant Conversion factor
            print('Constant value assumed for the M/L')
            ##-- Using a sigma clipping to find the real maximum
            maxVobs = np.max(sig_clip(self.Vcobs_int, 5.), axis=None)
            maxVfield = np.max(sig_clip(Vfield[np.where(self.rpc < Vcobs_rintpc[-1])], 5.), axis=None)

            ##-- Isolate the 25% highest values and get the mean of that as a scaling measure
            meanVobs = np.mean(self.Vcobs_int[
                               np.where(self.Vcobs_int / maxVobs > 6./10.)], axis=None)
            meanVfield = np.mean(Vfield[np.where(Vfield/maxVfield > 6/10.)], axis=None)
            ConstFactor = meanVobs / meanVfield

            VFactor[:,:] = ConstFactor
            Vfactor_profile[:] = ConstFactor
            for i in range(len(Vcobs_rintpc)-1):
                sel = np.where((self.rpc >= Vcobs_rintpc[i]) & (self.rpc < Vcobs_rintpc[i+1])) 
                if len(sel) > 0 :
                    ##-- Average of the Vfield within the annulus
                    Vmodel_unscaled[i] = np.mean(Vfield[sel], axis=None)
                
        ##-- Imposing the last value 
        Vmodel_unscaled[-1] = Vmodel_unscaled[-2]

        ##--- M/L is scaled by the square of the V ratio
        self.ML = VFactor**2.0
        
        self.Vmodel_unscaled = Vmodel_unscaled
        self.Vfactor_profile = Vfactor_profile
        if self.plot:
            p.clf()
            # Create plot of radial profile
            p.plot(self.Vcobs_r, self.Vcobs, 'bo')         
            # Error bars on observed Vc
            p.errorbar(self.Vcobs_r, self.Vcobs, 
                    yerr=self.eVcobs, ecolor='b') 
            # Overplot of Input Vc
            p.plot(self.Vcobs_rint, Vmodel_unscaled * Vfactor_profile, 
                    '--r')  
            p.xlabel("R [pc]")
            p.ylabel(r"$V_c$ [km/s]")
            p.ylim(0., maxVcobs * 1.3)
            p.draw()
            if stop_program() : sys.exit(0)
            
    #==============================================================
    #---------------Potential decomposition------------------------
    #==============================================================
    def decompose(self, pot, Nbins, fc_order=8):
          
        if self.verbose :
            print("Decomposition of the potential in Fourier coefficients... \n")         
        r = self.rpc
        theta = self.theta
        maxr = np.max(r, axis=None)/2
        stepr = maxr / (Nbins-1)
        x = np.arange(0., maxr+stepr, stepr)
        y = np.zeros_like(x)
        ##Definition of Fouriercoefficients:
        fc_cos = np.zeros((Nbins, fc_order))
        fc_sin = np.zeros_like(fc_cos)
        fc_cos_n = np.zeros_like(fc_cos)
        fc_sin_n = np.zeros_like(fc_cos)
        #fft_res = np.zeros((len(x),len(x)))
        ##== Filling in the values for y (only if there are some selected pixels)
        for i in range(len(x)-1):
            sel = np.where((r >= x[i]) & (r < x[i+1]))  ##== selecting an annulus between two bins
            if len(sel) > 0 :
                for j in range(fc_order):
                    fc_cos[i,j] = np.mean(np.cos(j*theta[sel]) * pot[sel], axis=None)
                    fc_sin[i,j] = np.mean(np.sin(j*theta[sel]) * pot[sel], axis=None)
                fc_cos_n[i,:]= fc_cos[i,:]/(np.mean(pot[sel], axis=None)) 
                fc_sin_n[i,:]= fc_sin[i,:]/(np.mean(pot[sel], axis=None)) 
                             
        self.fc_cos = fc_cos
        self.fc_sin = fc_sin
        self.fc_cos_n = fc_cos_n
        self.fc_sin_n = fc_sin_n
                         
          
    #============================================================
    #----------------Calculation of the forces-------------------
    #============================================================
    def calc_force(self):

        if self.verbose :
            print("Calculating the forces ... \n")           
        F_grad = np.gradient(self.pot) #mass of the probe set to 1
        self.Fx = F_grad[1] / (self.steparc * self.pc_per_arcsec)
        self.Fy = F_grad[0] / (self.steparc * self.pc_per_arcsec)

        # F_morph_grad = morphological_gradient(pot,size=(10,10)) 

        theta = self.theta 
        #Radial force vector in outward direction
        self.Frad =   self.Fx * np.cos(theta) + self.Fy * np.sin(theta)
        #Tangential force vector in clockwise direction
        self.Ftan = - self.Fx * np.sin(theta) + self.Fy * np.cos(theta)
        
    #============================================================
    #----Calculation of the gravity torques and mass flow rates--
    #============================================================
    def calc_torque(self, r, v, Fx_grad, Fy_grad, Nbins):
              
        if self.verbose :
            print("Calculating the torques... \n")
        self.torque = (self.xpc * Fy_grad - self.ypc * Fx_grad) * self.gasdisk_dep
        ##Average over azimuthal angle and normalization
        maxr = np.max(r, axis=None)/2
        stepr = maxr / (Nbins-1)
        rsamp = np.arange(0., maxr+stepr, stepr)
        t = np.zeros_like(rsamp)
        l = np.zeros_like(rsamp)
        dl = np.zeros_like(rsamp)
        dm = np.zeros_like(rsamp)
        dm_sum = np.zeros_like(rsamp)
        ##== Filling in the values for y (only if there are some selected pixels)
        for i in range(len(rsamp)-1):
            sel = np.where((r >= rsamp[i]) & (r < rsamp[i+1]))  ##== selecting an annulus between two bins
            if len(sel) > 0 :
                ##Torque per unit mass averaged over the azimuth:
                t[i] = np.mean(self.torque[sel], axis=None) / np.mean(self.gasdisk_dep[sel], axis=None)
                ## Angular momentum averaged over the azimuth:
                l[i] = np.mean(r[sel]) * np.mean(v[sel])
                ##Specific angular momentum in one rotation
                dl[i] = t[i] / l[i] * 1.0   
                ##Mass inflow/outflow rate as function of radius( dM/(dR dt) ):
                dm[i] = dl[i] * 2*np.pi * np.mean(r[sel]) * np.mean(self.gasdisk_dep[sel])
                ##Inflow/outflow rates integrated out to a certain radius R
                dm_sum[i] = np.sum(dm[0:i]) * stepr
        self.torque_r = rsamp
        self.torque_mean = t
        self.torque_l = l
        self.torque_dl = dl
        self.torque_dm = dm
        self.torque_dm_sum = dm_sum
