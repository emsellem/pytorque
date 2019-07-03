#!/usr/bin/python

"""
    Example to run pypot with an image from NGC 7217
"""

##== Properties for the galaxy
Distdef = 16.0        # Galaxy distance in Mpc
PAdef=268.0          # Position Angle
Incldef=33.7          # Inclination

"""
Import of the required modules
"""
import pypot
from pypot.pypot_main import *
import pylab as p
import numpy as num
from scipy.ndimage.interpolation import rotate

dir_input = "data_7217"

def pypot_example(verbose='on', plotlevel=2, debug='on'):
    #####################################################################################################
    # Here we begin the action
    # by calling the class/functions
    #####################################################################################################
    gal = galaxymodel(dir=dir_input, inimage="NGC_7217:I:H:efp2002.fits", gasimage="7217cd-dm0.fits", rotfile="rot-7217cd3_both.txt", verbose=verbose, plotlevel=plotlevel, debug=debug, dist=Distdef, PA=PAdef, Incl=Incldef, cen=(158,149), cengas=(512,512), box=(90,90), boxgas=(60,60), pixelsize=1.5, pixelsizegas=2.0, azoom=4, Nbins=90)
    
    ## background then deprojection 
    gal.sub_background()
    gal.deproj_disk()
    ## Radial Profil for Bulge (exclude opening angle around major axis)
    gal.radial_excl(gal.rpc, gal.diskS, gal.Nbins)
    ## Bulge fit
    gal.bulge_fit(gal.diskS)
    ## Substraction of bulge from projected disk 
    gal.bulge_sub(gal.diskN)
    ## Deprojection of disk without bulge
    gal.deproj_disk_nobulge()
    ## Adding bulge component again to deprojected disk
    gal.bulge_add(gal.diskS_nobulge)
    ##Radial Profile
    rdiskpc, disk = gal.comp_profile(gal.rpc, gal.diskF, gal.Nbins)
    ## Deriving the kernel
    gal.calc_kernel(softparam=0.)
    ##Calculation of potential
    gal.calc_pot()
    ##Calculation of forces (gradient of potential) - unscaled
    gal.calc_force(gal.pot)
    ##Create rotation velocity field, unscaled here - from force field
    gal.VcU = gal.calc_vrot_field(gal.Frad)
    ##Create rotation velocity profile, unscaled here
    gal.rsampVc, gal.sampVcU = gal.comp_profile(gal.rpc, gal.VcU, Nbins=gal.Nbins)
    ## Calculation of radial dependent M to L ratio by comparison with observed velocity
    ## mode 1: radial dependent, else constant scaling value
    gal.scale_Vc(gal.file_Vcobs, gal.VcU, mode=0)
    ##-- We can now scale the potential and forces
    gal.potC = gal.ML * gal.pot
    gal.FxC = gal.ML * gal.Fx
    gal.FyC = gal.ML * gal.Fy
    gal.FradC = gal.ML * gal.Frad
    gal.FtanC = gal.ML * gal.Ftan
    gal.VcC = num.sqrt(gal.ML) * gal.VcU
    #Decomposition of potential
    gal.decompose(gal.potC, Nbins=45, fc_order=8)
    ##Calculation of torques
    gal.calc_torque(gal.rpc, gal.VcC, gal.FxC, gal.FyC, Nbins=gal.Nbins)
    
    ## plotting the result
    if gal.plotlevel >= 1 :
        p.clf()
        p.plot(gal.torque_r,gal.torque_dm_sum)  # Create plot of radialprofile
        p.xlabel("R")
        p.ylabel("Torque")
        p.draw()
        ok = input("Type any key to continue")

    p.clf()
    vmax = num.median(num.ravel(gal.torque)) + num.std(gal.torque, axis=None) * 5
    vmin = num.median(num.ravel(gal.torque)) - num.std(gal.torque, axis=None) * 5
    im = p.imshow(rotate(gal.torque, 90.-gal.PA), vmin=vmin, vmax=vmax)
    p.colorbar()
    p.title("Radial Torques")
    p.draw()

    return gal
