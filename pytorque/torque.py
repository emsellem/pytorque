#!/usr/bin/python

"""
Program for computing the gravitational potential from a combination 
of optical and near-infrared images. 
MPIA / CRAL: @ 2007

Authors: Eric Emsellem for first python modules and new class adaptation.
         With great contributions from Sebastian Haan
"""

__version__ = '1.0.0 (05-11, 2019)'

## Changes -- 
##   05/11/19- EE - v1.0.0: A bit of cleaning and docs
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
from astropy.stats import sigma_clip

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

## Importing all function from the pytorque module
from misc_functions import *

##############################################################
######### GALAXYMODEL CLASS ##################################
##############################################################
class discmodel(object)   :
    """Galaxy model class
    Attributes
    ----------
    folder : str
        file folder where the input images are
    mass_filename : str
        Name of Mass image (fits)
    gas_filename :  str
        Name of gas distribution image (fits)
    vc_filename : str
        rotation file
    distance : float
        Distance of galaxy in Mpc [1.]
    inclination : float
        Inclination of galaxy in degrees [0.]
    cen: tuple of floats
        centre of mass image (in pixel)
    cengas: tuple of floats
        centre of gas image (in pixel)
    box: tuple of floats
        size of mass image box to extract (in pixel)
    boxgas: tuple of floats
        size of gas image box to extract (in pixel)
    pixelsize: float [1.]
        Size of pixel in arcseconds for mass image
    pixelsizegas: float [1.]
        Size of pixel in arcseconds for gas image
    azoom: int [6]
        Zooming factor. Should be larger or equal to 1.
    plot: bool
        Whether to plot or not [True]
    verbose : bool
        Verbose option [False]
    """
    def __init__(self, folder="", mass_filename=None, gas_filename=None, 
                 vc_filename=None, vc_filetype="ROTCUR",
                 verbose=False, distance=1., PA=0., inclination=0., 
                 cen=(512,512), cengas=(512,512), 
                 box=(300,300), boxgas=(300,300), 
                 step_arc=1., stepgas_arc=1., azoom=1, 
                 plot=True, n_rbins=100):
        """Initialisation of the disc model class
        Mainly setting up a few parameters like the distance, PA and inclination
        and reading the data from the given input images

        Attributes
        ----------
        folder : str
            file folder where the input images are
        mass_filename : str
            Name of Mass image (fits)
        gas_filename :  str
            Name of gas distribution image (fits)
        vc_filename : str
            rotation file
        distance : float
            Distance of galaxy in Mpc [1.]
        PA: float
            Position Angle in degrees [0.] of the line of nodes
            from North counter-clockwise.
        inclination : float
            Inclination of galaxy in degrees [0.]
        cen: tuple of floats
            centre of mass image (in pixel)
        cengas: tuple of floats
            centre of gas image (in pixel)
        box: tuple of floats
            size of mass image box to extract (in pixel)
        boxgas: tuple of floats
            size of gas image box to extract (in pixel)
        step_arc: float [1.]
            Size of pixel in arcseconds for mass image
        stepgas_arc: float [1.]
            Size of pixel in arcseconds for gas image
        azoom: int [6]
            Zooming factor. Should be larger or equal to 1.
        n_rbins: int [100]
            Number of radial bins for profiles
        plot: bool
            Whether to plot or not [True]
        verbose : bool
            Verbose option [False]
        """

        ##--- Verbose and plot option - self explanatory
        self.verbose = verbose
        self.plot = plot

        ##=== Some useful number
        self.distance = distance                            # Galaxy distance in Mpc
        self.pc_per_arcsec = self.distance * np.pi / 0.648  # Conversion arcsec => pc (pc per arcsec)
        self.PA = PA                                        # Position angle in degrees (from North)
        self.inclination = inclination                      # Inclination in degrees

        ##== Checking the existence of the folder 
        if not os.path.exists(folder):
            print('ERROR: Input Path {0} does not exists, sorry!'.format(
                    folder))
            return
        self.folder = folder 

        #--------------- Input frame characteristics --------------------
        self.cen = cen                  # Position for the centre
        self.box = box                  # size of the box (total) in pixels  
        self.cengas = cengas            # Position (pixels) for the centre of gas image
        self.boxgas = boxgas            # Size of gas box (pixels)
        self.step_arc = step_arc
        self.stepgas_arc = stepgas_arc

        #--- Number of radial bins for profiles
        self.n_rbins = n_rbins              

        # Mass image name -if any
        self.mass_filename = mass_filename
        # Gas image name
        self.gas_filename = gas_filename
        # Vcfile name and type
        self.vc_filename = vc_filename
        self.vc_filetype = vc_filetype

        self._found_mass = self._found_gas = self._found_vc = False
        if mass_filename is None :
            print('WARNING: no filename for the main Input Image provided')
        if gas_filename is None :
            print('WARNING: no filename for the gas Input Image provided')
        if vc_filename is None :
            print('WARNING: no filename for the Rotation Input File provided')

        ##== Checking if the zoom is larger than 1 before proceeding
        if azoom >= 1 :
            self.azoom = azoom
            if self.verbose:
                print("Zoom will be {0:d}".format(self.azoom)) 
        else :
            print("ERROR: zoom factor (azoom) should be larger than 1!")
            return

        # Initialisation of the model
        self.init_model()

        # initialisation of the arrays
        self.init_discs()

    def init_model(self):
        """Initialise the model by reading the Mass, gas and Vc files
        """
        ##== We now save these information into the class
        if self.verbose:
            print("Trying to open Input files now...")

        # Full Mass/Gas/Vc image name
        if self.mass_filename is not None:
            inimage_name = joinpath(self.folder, self.mass_filename)
            
            # Extracting the data from Mass Image
            self.data, h, self.step_arc = extract_frame(inimage_name, self.step_arc, self.verbose)
            # Step in parsec - Mass
            self.steppc = self.step_arc * self.pc_per_arcsec   # Pixel size in pc
            self._found_mass = (self.data is not None)

        # Extracting the data from Gas Image
        if self.gas_filename is not None:
            full_gas_filename = joinpath(self.folder, self.gas_filename)
            self.data_gas, h, self.stepgas_arc = extract_frame(full_gas_filename, 
                                    self.stepgas_arc, self.verbose)
            # Step in parsec - Gas
            self.steppc2 = self.stepgas_arc * self.pc_per_arcsec # Pixel size of gas image in pc
            self.zoomfactor = self.stepgas_arc / self.step_arc
            self._found_gas = (self.data_gas is not None)
            if self.verbose:
                print(("zoomfactor (pixel size ratio) = {0}".format(self.zoomfactor)))

        if self.read_Vc() < 0:
            print("ERROR: reading Vc file did not succeed")
        else:
            self._found_vc = True
    
    def run_torques(self, softening=0):
        """Run the series of processes needed to calculate
        the torques 

        Input
        -----
        softening: float
            Softening applied to the torques. The softening
            is applied to the kernel.
        """
        self.sub_background()
        self.deproject_disc()
        self.radial_excl(self.rpc, self.discS, self.n_rbins)

        # Bulge fitting
        self.bulge_fit(self.discS)
        ## Substraction of bulge from projected disc 
        self.bulge_sub(self.discN)

        ## Deprojection of disc without bulge
        self.deproject_disc_nobulge()
        ## Adding bulge component again to deprojected disc
        self.bulge_add(self.discS_nobulge)

        ## Radial Profile
        self.rdiscpc, self.prof_disc = extract_profile(self.rpc, self.discF,  self.n_rbins)

        ## Deriving the kernel
        self.calc_kernel(softening=softening)
        ## Calculation of potential
        self.calc_pot()
        ## Calculation of forces (gradient of potential) - unscaled
        self.calc_force()
        ## Create rotation velocity field, unscaled here - from force field
        self.VcU = self.calc_vrot_field()

        ## Create rotation velocity profile, unscaled here
        self.rsampVc, self.sampVcU = extract_profile(self.rpc, self.VcU, n_rbins=self.n_rbins)
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
        self.decompose(self.potC, n_rbins=45, fc_order=8)

        ##Calculation of torques
        self.calc_torque(self.rpc, self.VcC, self.FxC, self.FyC, self.n_rbins)
        
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
        
    def init_discs(self):
        """Initialise both stars and gas discs
        """
        self.init_disc()
        self.init_gasdisc()

    def init_disc(self):
        """Initialise the array for the disc
        """
        Xwidth = self.box[0] 
        Ywidth = self.box[1]
        self.box_z = np.array([Xwidth * self.azoom, Ywidth * self.azoom]) ##== Zoomed box size
        ##== Total size will be twice this
        self.Xtotdisc, self.Ytotdisc = self.box_z[0]*2, self.box_z[1]*2
        ##== And defining the new center
        self.xcen_z, self.ycen_z = self.box_z[0], self.box_z[1]   
        ##== Allocating the right space
        self.disc = np.zeros((self.Xtotdisc, self.Ytotdisc))  
        if self.verbose:
            print("New Center = {0:d}, {1:d} - "
                  "Box size = {2:d}, {3:d}".format(
                      self.xcen_z, self.ycen_z, 
                      self.Xtotdisc, self.Ytotdisc))

        self.Rdiscpx = self.box[0] / 2.0     

        ##== Getting the data within the defined box
        ##== Note that array has length =  2 times Xwidth (resp. Ywidth) and the center at Xwidth (Ywidth)
        ##== We are therefore only filling the inner part of the new array with the original data

        ## Useful (non zero) original part of the data
        self.usediscX = [self.xcen_z - Xwidth, self.xcen_z + Xwidth]
        self.usediscY = [self.ycen_z - Ywidth, self.ycen_z + Ywidth]

        if self._found_mass:
            self.disc[self.usediscX[0]:self.usediscX[1],
                      self.usediscY[0]:self.usediscY[1]] = \
                              self.data[self.cen[1] - Xwidth: self.cen[1] + Xwidth,
                                        self.cen[0] - Ywidth: self.cen[0] + Ywidth] 
            data = self.data
        else:
            data = np.zeros_like(self.disc)

        if self.plot: 
            p.figure(1)
            p.clf()
            p.imshow(np.log10(data))
            p.title("Original Imaging data")
            p.figure(2)
            p.imshow(np.log10(self.disc))
            p.title("Transfered data after zooming/box")
            p.draw()
            if stop_program() : sys.exit(0)
            p.close(2)

    def init_gasdisc(self):
        """Initialise gas disc
        """
        if not hasattr(self, 'disc'):
            print("ERROR: disc was not initialised. First do this with init_disc")
            return

        self.gasdisc = np.zeros_like(self.disc)

        ## Converting cellsize of GAS array to same cell-size as Input Main Image array#
        if self._found_gas:
            data_gas = self.data_gas
        else:
            data_gas = np.zeros_like(self.data)
            self.zoomfactor = 1

        self.data_gas_z = zoom(data_gas, self.zoomfactor)
        if self.plot:
            p.figure(1)
            p.clf()
            p.imshow(np.log10(data_gas))
            p.title("Original gas data")
            p.figure(2)
            p.imshow(np.log10(self.data_gas_z))
            p.title("Zoomed Gas data")
            p.draw()
            if stop_program() : sys.exit(0)
            p.close(2)
        xcen_zoom = np.round(self.cengas[0] * self.zoomfactor).astype(np.int)
        ycen_zoom = np.round(self.cengas[1] * self.zoomfactor).astype(np.int)
        self.boxgas_z = np.array([np.round(self.boxgas[0] * self.zoomfactor).astype(np.int),
                                  np.round(self.boxgas[1] * self.zoomfactor).astype(np.int)])

        ##== Now taking the right width to extract the gas info
        if self.verbose:
            print("Rescaled Gas data had +/- {0}, {1} pixels".format(self.boxgas_z[0]*2, self.boxgas_z[1]*2))
            print("To be included in +/- {0}, {1} pixels".format(self.box_z[0]*2, self.box_z[1]*2))
        minXwidth = np.minimum(self.boxgas_z[0], self.box_z[0])
        minYwidth = np.minimum(self.boxgas_z[1], self.box_z[1])

        ## Specifying the array for gas distribution
        if self.verbose:
            print("Pixel Coordinates for the gas data")
            print(self.ycen_z-minYwidth, self.ycen_z+minYwidth, 
                    self.xcen_z-minXwidth, self.xcen_z+minXwidth)
            print("Corresponding pixels in new Gas array")
            print(ycen_zoom-minYwidth, ycen_zoom+minYwidth, 
                    xcen_zoom-minXwidth, xcen_zoom+minXwidth)
        self.gasdisc[self.ycen_z-minYwidth:self.ycen_z+minYwidth, 
                     self.xcen_z-minXwidth:self.xcen_z+minXwidth] = \
                             self.data_gas_z[ycen_zoom-minYwidth:ycen_zoom+minYwidth, 
                                             xcen_zoom-minXwidth:xcen_zoom+minXwidth]
        
        ## Simple Deprojecting of gas disc (including rotation)
        self.gasdisc_dep = deproject_frame(self.gasdisc, self.PA, self.inclination)
        if self.plot :
            p.figure(1)
            p.clf()
            p.imshow(np.log10(self.gasdisc))
            p.title("Zoomed gas data")
            p.figure(2)
            p.imshow(np.log10(self.gasdisc_dep))
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
        self.background = np.mean(self.disc[self.usediscY[0]:self.usediscY[0]+20,
                                            self.usediscX[0]:self.usediscX[0]+20], 
                                            axis=None) 
        if self.verbose :
             print("Background value found to be {0}".format(self.background))
             print("Subtracting this background value... \n")

        self.discN = self.disc - self.background

        ##-- Replacing all values below 0. with 0.
        self.discN[np.where(self.discN < 0.)] = 0.

    #============================================================
    #--Deprojecting of the disc (includes rotation of the major axis to the north axis)--
    #============================================================
    def deproject_disc(self) :
        """ Deproject the data using the normalised disc intensity, the PA and
             inclination
        """
        if self.verbose :
            print("Deprojecting the disc ... \n")
        self.discS = deproject_frame(self.discN, self.PA, self.inclination)
        if self.plot :
            p.figure(1)
            p.clf()
            p.imshow(np.log10(self.discS))
            p.title("Deprojected disc")
            p.figure(2)
            p.imshow(np.log10(self.discN))
            p.title("Original disc (bulge subtracted)")
            p.draw()
            if stop_program() : sys.exit(0)
            p.close(2)

        ##== Calculate Cartesian coordinates x,y in pc in respect to the center (coord:0,0):
        ##== No central pixel
        xpix, ypix = self.discS.shape
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
    def radial_excl(self, r, data, n_rbins, wedge_size=30.0):   
        """Create radial profile after excluding a wedge defined
        by an angle

        Input
        -----
        r: float array
            Input radius array
        data: float array
            Input data array
        n_rbins: int
            Number of bins for the radial profile
        wedge_size: float
            Wedge size in degrees. Will exclude wedge with size
            +/-wedge_size on both sides of the major-axis.

        Returns
        -------
        rad: array
            Radial 1D array for the radii
        values: array
            Radial 1D array for the values
        """
        if self.verbose :
            print("Deriving the radial profile for bulge... \n")

        ##== Adding 1/2 step
        rsamp, step = get_rsamp(r, n_rbins)
        rdata = np.zeros_like(rsamp)

        ##== Wedge in radians
        wedge_rad = np.deg2rad(wedge_size)

        ##== Filling in the values for y (only if there are some selected pixels)
        for i in range(len(rsamp)-1):
            ##== selecting an annulus between two bins
            ##== and excluding the wedge +/- along the major-axis
            sel = np.where((r >= rsamp[i]) & (r < rsamp[i+1]) 
                    & (((self.theta > wedge_rad) 
                       & (self.theta < np.pi - wedge_rad)) 
                    | ((self.theta > np.pi + wedge_rad) 
                       & (self.theta < 2. * np.pi - wedge_rad))))  
            if len(sel) > 0 :
                rdata[i] = np.mean(data[sel], axis=None)

        ##== Saving this into the discmodel class
        self.rbin_excl = rsamp
        self.rmean_excl = rdata
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
    def bulge_fit(self, disc):
            
        if self.verbose :
            print("Fitting the disc and the bulge... \n")
        Xwidth, Ywidth = self.box[0] * self.steppc, self.box[1] * self.steppc
        x = self.rbin_excl
        ind = np.where(x < Xwidth)
        x = x[ind]
        y_meas = self.rmean_excl[ind]
        I_max = np.max(disc, axis=None)
        Imax_bulge, Rbulge, Imax_disc, Rdisc = I_max*0.5, 0.02*Xwidth, I_max*0.5, 0.4*Xwidth  #Initial values
        #I_bulge = Imax_bulge * pow((1+(x/Rbulge)**2),(-5/2)) #Rgo Plummer Sphere
        I_bulge = Imax_bulge * 4/3 * Rbulge**5 * pow((x**2 + Rbulge**2),(-2)) #projected Plummer Sphere
        I_disc  = Imax_disc * np.exp(-x/Rdisc)  #Model for disc
        I_disc  = Imax_disc * Rdisc / (2*np.pi) * pow((x**2 + Rdisc**2),(-3/2)) #Kuzmin disc
        y_true = I_bulge + I_disc     #sum of both
                  
        def residuals(par, y, x):           # create residual equation: y_true - y_meas
            Imax_bulge, Rbulge, Imax_disc, Rdisc = par
            #err = y - (Imax_bulge * pow((1+(x/Rbulge)**2),(-5/2)) + Imax_disc * np.exp(-x/Rdisc))
            err = y - ( Imax_bulge * Rbulge**5 * 4/3 * pow((x**2 + Rbulge**2),(-2)) + Imax_disc * np.exp(-x/Rdisc) )
            return err
                  
        def peval_all(x, par):
            #return par[0] * pow((1+(x/par[1])**2),(-5/2)) + par[2] * np.exp(-x/par[3])
            return par[0] * par[1]**5 * 4/3 * pow((x**2 + (par[1])**2),(-2)) + par[2] * np.exp(-x/par[3])
        
        def peval_bulge(x, par):
            #return par[0] * pow((1+(x/par[1])**2),(-5/2))
            return par[0] *  par[1]**5 * 4/3 * pow((x**2 + (par[1])**2),(-2))
        
        def peval_disc(x, par):
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
            line = p.plot(x,np.log10(y_meas+1.e-2), 'k',x,np.log10(peval_bulge(x,self.plsq[0])+1.e-2),'b', x,np.log10(peval_disc(x,self.plsq[0])+1.e-2), 'r',) #,r_bulge,bulge_mean)
            p.legend(line, ("Total", "Bulge", "Disc"))
            p.xlabel("R [pc]")
            p.ylabel("Flux, Bulge, Disc (log10)")
            p.draw()
            if stop_program() : sys.exit(0)
          
    def bulge_sub(self, disc):
        if self.verbose :
            print("Subtracting the bulge component... \n")
        self.disc_nobulge = disc - self.bulge
          
    def bulge_add(self, disc):
        if self.verbose :
            print("Adding the bulge to the deprojected disc ... \n")
        self.discF = disc + self.bulge

    def deproject_disc_nobulge(self) :
        if self.verbose :
            print("Deprojecting the disc without bulge... \n")
        self.discS_nobulge = deproject_frame(self.disc_nobulge, self.PA, self.inclination)
          
#============================================================
#----------------Calculation of the Kernel Function----------
#============================================================
    def calc_kernel(self, softening=0., function="sech2"):                       
        """Calculate the kernel for the potential
        
        Input
        -----
        softening: float
            Size of softening in pc
        function: str
            Name of function for the vertical profile
            Can be sech or sech2. If not recognised, will use sech2.

        """
        if self.verbose :
            print("Deriving the kernel ... \n")
        self.softening = softening                     # Softening Parameter
        self.hzpx = np.int(self.Rdiscpx / 12.)         # In grid size, the vertical height is taken as 1/12 of the disc radius

        ##== Grid in z from -hz to +hz with no central point
        self.zpx = np.arange(0.5 - self.hzpx, self.hzpx + 0.5, 1.) 
        self.zpc = self.zpx * self.steppc              # Z grid in pc

        if function == "sech" :                        # sech or sech2 function (1/cosh = sech)                                  
            self.h = sech(self.zpx / self.hzpx)        # Vertical thickening function in z, has to be normalized later 
        elif function == "sech2" :
            self.h = (sech(self.zpx / self.hzpx))**2       
        else :
            self.h = (sech(self.zpx / self.hzpx))**2     

        Sumh = np.sum(self.h, axis=None)              # Integral of h over the entire range
        self.hn = self.h / Sumh                               # Normalised function h

        kernel = self.hn[np.newaxis,np.newaxis,...] / (np.sqrt(self.rpc[...,np.newaxis]**2 + self.softening**2 + self.zpc**2)) 
        self.kernel = np.sum(kernel, axis=2)

    #============================================================
    #----------------Calculation of the potential----------
    #============================================================
    def calc_pot(self):
        """Calculate the potential (self.pot) given the convolution kernel
        The potential is just : G * Mass_density * convolution_kernel
        Using Fourier transform.

        """
        if self.verbose :
                  print("Calculating the potential ... \n")

        # Initialise array with zeroes
        self.convol = fftconvolve(self.kernel, self.discF, mode='same') # Convolution of the kernel with the weight (mass density)

        ##== Saving this into the discmodel class
        self.pot = -Ggrav.value * self.convol 

    #============================================================
    #-------- Calculation of Circular Velocity field ------------
    #============================================================
    def calc_vrot_field(self):
        """Calculate rotation velocities from Force field

        Returns: velocity map (float array)
        """
        if self.verbose :
            print("Calculating the rotation velocities from force field... \n")

        return np.sqrt(np.fabs(self.rpc * self.Frad))

    def read_Vc(self, vc_filename=None, vc_filetype="ROTCUR"):
        """Reading the input Vc file

        Input
        -----
        vc_filename: str
            Name of the Vcfile

        vc_filetype: str
            'ROTCUR' or 'ASCII'

        Returns
        -------
        status: int
            0 means it was read. -1: the file does not exist
            -2: file type not recognised.
        """
        if vc_filename is None:
            if self.vc_filename is None:
                print("ERROR: no Vc filename provided")
                return -1

            vc_filename = self.vc_filename
        if self.verbose :
            print("Reading the Vc file")

        ##--- Reading of observed rot velocities
        vc_filename = joinpath(self.folder + vc_filename)
        status, self.Vcobs_r, self.Vcobs, self.eVcobs, \
                self.Vcobs_rint, self.Vcobs_int = read_vcirc_file(vc_filename, 
                        vc_filetype=vc_filetype)

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
            maxVobs = np.max(sigma_clip(self.Vcobs_int, sigma=5.), axis=None)
            maxVfield = np.max(sigma_clip(Vfield[np.where(self.rpc < Vcobs_rintpc[-1])], sigma=5.), axis=None)

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
    def decompose(self, pot, n_rbins, fc_order=8):
          
        if self.verbose :
            print("Decomposition of the potential in Fourier coefficients... \n")         
        r = self.rpc
        theta = self.theta

        ## Getting the radial sampling
        x, stepr = get_rsamp(r, n_rbins)
        y = np.zeros_like(x)

        ##Definition of Fouriercoefficients:
        fc_cos = np.zeros((n_rbins, fc_order))
        fc_sin = np.zeros_like(fc_cos)
        fc_cos_n = np.zeros_like(fc_cos)
        fc_sin_n = np.zeros_like(fc_cos)

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
        """Calculation of the forces
        """
        if self.verbose :
            print("Calculating the forces ... \n")           

        # Force from the gradient of the potential
        F_grad = np.gradient(self.pot) 

        # Force components in X and Y
        self.Fx = F_grad[1] / (self.step_arc * self.pc_per_arcsec)
        self.Fy = F_grad[0] / (self.step_arc * self.pc_per_arcsec)

        #Radial force vector in outward direction
        self.Frad =   self.Fx * np.cos(self.theta) + self.Fy * np.sin(self.theta)
        #Tangential force vector in clockwise direction
        self.Ftan = - self.Fx * np.sin(self.theta) + self.Fy * np.cos(self.theta)
        
    #============================================================
    #----Calculation of the gravity torques and mass flow rates--
    #============================================================
    def calc_torque(self, r, v, Fx_grad, Fy_grad, n_rbins):
        """Calculation of the gravity torques
        """
        if self.verbose :
            print("Calculating the torques... \n")

        # Torque is just Deprojected_Gas * (X * Fy - y * Fx)
        self.torque = (self.xpc * Fy_grad - self.ypc * Fx_grad) * self.gasdisc_dep

        ## Average over azimuthal angle and normalization
        rsamp, stepr = get_rsamp(r, n_rbins)

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
                t[i] = np.mean(self.torque[sel], axis=None) / np.mean(self.gasdisc_dep[sel], axis=None)
                ## Angular momentum averaged over the azimuth:
                l[i] = np.mean(r[sel]) * np.mean(v[sel])
                ##Specific angular momentum in one rotation
                dl[i] = t[i] / l[i] * 1.0   
                ##Mass inflow/outflow rate as function of radius( dM/(dR dt) ):
                dm[i] = dl[i] * 2*np.pi * np.mean(r[sel]) * np.mean(self.gasdisc_dep[sel])
                ##Inflow/outflow rates integrated out to a certain radius R
                dm_sum[i] = np.sum(dm[0:i]) * stepr
        self.torque_r = rsamp
        self.torque_mean = t
        self.torque_l = l
        self.torque_dl = dl
        self.torque_dm = dm
        self.torque_dm_sum = dm_sum
