import pandas as pd
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import wcs_to_celestial_frame
import math
import os
import lacosmic
from scipy import optimize
from astroquery.vizier import Vizier 
import astropy.coordinates as coord
from line_profiler import LineProfiler
from astropy.stats import sigma_clip
from scipy import integrate
from astropy.table import Table,vstack
from photutils.datasets import (make_random_gaussians_table,
                                 make_noise_image,
                                 make_gaussian_sources_image)
from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from scipy.stats import sigmaclip

import Sub_PSF as CP

plt.close('all')

def magnitude(counts,exposure):
    mag =  2.5*np.log10(counts/exposure)
    return(mag)

def error(skyerror):
    err = np.sqrt(((2.5/np.log(10)) * np.sqrt(skyerror))**2 + 0.03**2 + 0.011**2)
    return(err)
    
def count_conversion(FluxI,ExpI,ZPI,ErrI,FluxT,ExpT,ZPT,ErrT):#Gives subtracted Mag
    #Need to convert the counts of template, and minus from the image
    #convert the flux of the template with a different ZP and Exposure
    #So ZPB and ExpB belong to the image
    #ZPA and ExpA belong the the template
    print('VALUES')
    print(FluxI,ExpI,ZPI,ErrI,FluxT,ExpT,ZPT,ErrT)
    term1 = (ZPI-ZPT) + 2.5*np.log10(FluxT/ExpT)
    print('TERM1 ',term1)
    term1_err = 2.5*ErrT/(FluxT*np.log(10))
    print('TERM1 ',term1_err)
    FluxB = ExpI * 10**((term1)/2.5)
    print('FLUXB ',FluxB)
    FluxB_err = FluxB * (1/2.5) * np.log(10) * term1_err 
    print('FLUXB ',FluxB_err)
    print('TEMP MAG')
    print(FluxB,FluxI)
    print('DIFF')
    print(FluxI-FluxB)
    Flux_Sub_Err = np.sqrt(ErrI**2+FluxB_err**2)
    print('ERROR')
    print(Flux_Sub_Err)
    Flux_Sub = FluxI - FluxB
    Mag_Sub = 2.5*np.log10(Flux_Sub/ExpI)
    Error_Sub = error(Flux_Sub_Err/Flux_Sub)
    return(Mag_Sub,Error_Sub)

def Get_MAG(results_table,Templates,Filt):
    V = results_table[results_table['Filter'] == Filt]
    V_Temp = Templates[Templates['Filter'] == Filt]
    print(V_Temp)
    print(V)
    print(V['Error'])
    for i in range(len(V)):
        V['MAG'][i],V['Error'][i] = count_conversion(V['Count'][i],V['Exposure'][i],V['ZP'][i],V['Error'][i],V_Temp['Count'][0],V_Temp['Exposure'][0],V_Temp['ZP'][0],V_Temp['Error'][0])
    print(V)
    V = V[V['MAG'] != 0]
    V['MAG'] = V['ZP'] - V['MAG']
    return(V['MJD'],V['MAG'],V['Error'],V)
    
    
def fitting(directory, groupid, sky_coord, ja200_coord):

    results_table = Table({'MJD':[],'Filter':[],'Count':[],'Error':[],'Exposure':[],'ZP':[],'TEMP':[],'MAG':[]})
    for filename in os.listdir(directory):
        if filename != '.DS_Store':
            MJD,zp,count,exp,err,filter_,temp = CP.Return(directory+'/'+filename,groupid,sky_coord,(ja200_coord[0],ja200_coord[1]))
            results_table.add_row([MJD,filter_,count,err,exp,zp,temp,0])
    
    """
    results_table = results_table.group_by(['MJD','Filter','TEMP'])
    results_table = results_table.groups.aggregate(np.mean)
    """

    
    results_table['MAG'] = results_table['ZP']-2.5*np.log10(results_table['Count']/results_table['Exposure'])
    Templates = results_table[results_table['TEMP'] == 1]
    print('TEMP')
    results_table = results_table[results_table['TEMP'] != 1]

    
   
    MJD,Mags,Error,B = Get_MAG(results_table,Templates,1)
    MJD,Mags,Error = zip(*sorted(zip(MJD,Mags,Error)))
    scatter = plt.scatter(MJD,Mags,label = 'B')
    plt.errorbar(MJD,Mags, yerr = Error, fmt='o')

    MJD,Mags,Error,V = Get_MAG(results_table,Templates,2)
    MJD,Mags,Error = zip(*sorted(zip(MJD,Mags,Error)))
    plt.scatter(MJD,Mags,label = 'V')
    plt.errorbar(MJD,Mags, yerr = Error, fmt='o')
    
    tab = vstack([B, V])
    
    MJD,Mags,Error,gp = Get_MAG(results_table,Templates,3)
    MJD,Mags,Error = zip(*sorted(zip(MJD,Mags,Error)))
    plt.scatter(MJD,Mags,label = 'gp')
    plt.errorbar(MJD,Mags, yerr = Error, fmt='o')
      
    tab = vstack([tab, gp])
    
    MJD,Mags,Error,ip = Get_MAG(results_table,Templates,4)
    MJD,Mags,Error = zip(*sorted(zip(MJD,Mags,Error)))
    plt.scatter(MJD,Mags,label = 'ip')
    plt.errorbar(MJD,Mags, yerr = Error, fmt='o')
    
    tab = vstack([tab, ip])
    
       
    MJD,Mags,Error,rp = Get_MAG(results_table,Templates,5)
    MJD,Mags,Error = zip(*sorted(zip(MJD,Mags,Error)))
    plt.scatter(MJD,Mags,label = 'rp')
    plt.errorbar(MJD,Mags, yerr = Error, fmt='o')
        
    tab = vstack([tab, rp])
      
    MJD,Mags,Error,SDSSI = Get_MAG(results_table,Templates,6)
    MJD,Mags,Error = zip(*sorted(zip(MJD,Mags,Error)))
    plt.scatter(MJD,Mags,label = 'SDSS-I')
    plt.errorbar(MJD,Mags, yerr = Error, fmt='o')
    
    tab = vstack([tab, SDSSI]) 
    
    MJD,Mags,Error,SDSSR = Get_MAG(results_table,Templates,7)
    MJD,Mags,Error = zip(*sorted(zip(MJD,Mags,Error)))
    plt.scatter(MJD,Mags,label = 'SDSS-R')
    plt.errorbar(MJD,Mags, yerr = Error, fmt='o')
    
    tab = vstack([tab, SDSSR])
         
    MJD,Mags,Error,SDSSZ = Get_MAG(results_table,Templates,8)
    MJD,Mags,Error = zip(*sorted(zip(MJD,Mags,Error)))
    plt.scatter(MJD,Mags,label = 'SDSS-Z')
    plt.errorbar(MJD,Mags, yerr = Error, fmt='o')

    tab = vstack([tab, SDSSZ])

    MJD,Mags,Error,SDSSG = Get_MAG(results_table,Templates,9)
    MJD,Mags,Error = zip(*sorted(zip(MJD,Mags,Error)))
    plt.scatter(MJD,Mags,label = 'SDSS-G')
    plt.errorbar(MJD,Mags, yerr = Error, fmt='o')

    tab = vstack([tab, SDSSG])
        
    ax = scatter.axes
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel('MJD')
    plt.ylabel('Magnitude')
    plt.savefig('Light_curve.png')
    plt.show()
    
    
    
    tab = tab.to_pandas()    
    tab = tab.drop(columns = ['TEMP'])
    tab.to_excel("Photometry_data.xlsx",sheet_name='Sheet_name_1') 
    
    return()

"""
#Plot together with ATLAS and SWIFT? (bit messy)
import SWIFT
import ATLAS   
ATLAS.ATLAS(full)   
SWIFT.plot(bmag3,'bmag')
SWIFT.plot(m2mag3,'m2mag') 
"""

#import ATLAS   
#ATLAS.ATLAS() 
# 
#fitting('/Users/eleonoraparrag/documents/Python/LCO_combine/ORIG/', 58670)
#sky_coord = '21:00:20.930 -21:20:36.06'
#ja200_coord =315.08720833 -21.34335


directory = input('Which directory am I looking in? ')
groupid = input('Please enter the object name or groupid as in the fits header ')
sky_coord = input('Please enter the Galactic coordinates in format "RA DEC" ')
ja200_coord = input('Please enter the ja200 coordinates in "RA DEC" ')
fitting(directory,groupid,sky_coord,ja200_coord.split())



#sky_coord = '02:26:18.55 -09:50:09.0'
#ja200_coord = (36.57729167, -9.83583333)
#/Users/eleonoraparrag/documents/Python/SN2019muj/SN_other/'
