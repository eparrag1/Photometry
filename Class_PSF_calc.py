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
from astropy.table import Table
from photutils.datasets import (make_random_gaussians_table,
                                 make_noise_image,
                                 make_gaussian_sources_image)
from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from scipy.stats import sigmaclip

import Class_PSF as CP

plt.close('all')


def fitting(directory, groupid, sky_coord, ja200_coord, cutoff = None):

    results_table = Table({'MJD':[],'Filter':[],'Mag':[],'Error':[],'ZP':[],'SUB':[]})
    for filename in os.listdir(directory):
        if filename != '.DS_Store':
            MJD,mg,zp,err,filter_,sub = CP.Return(directory+'/'+filename,groupid,sky_coord,(ja200_coord[0],ja200_coord[1]),cutoff)
            if isinstance(mg, float) and mg != 0.0:
                pass
            else:
                continue
            results_table.add_row([MJD,filter_,mg,err,zp,sub])

    results_table = results_table.group_by(['MJD','Filter','SUB'])
    results_table = results_table.groups.aggregate(np.mean)
    
    df = results_table.to_pandas()

    indices = (df[df['ZP'] == 0].index.to_numpy())
    
    for i in indices:
        zp = np.array(df.loc[(df['MJD'] == df['MJD'][i])&(df['Filter'] == df['Filter'][i])&(df['ZP'] != 0)]['ZP'])
        zp_idx = (df[(df['MJD'] == df['MJD'][i])&(df['Filter'] == df['Filter'][i])&(df['ZP'] != 0)].index)
        if len(zp_idx) == 0:
            df=df.drop([i])
        else:   
            df=df.drop([zp_idx[0]])
        df['ZP'][i] = zp
     
    if cutoff is not '':        
        df = df[(df['MJD'] < 58670) | (df['SUB'] == 1)]
    df['Mag'] = df['ZP'] - df['Mag']
    results_table = df
    
    results_table = results_table.drop(columns = ['SUB'])
    results_table.to_excel("Photometry_data.xlsx",sheet_name='Sheet_name_1') 
    
    B = results_table[results_table['Filter'] == 1]
    plt.errorbar(B['MJD'],B['Mag'], yerr = B['Error'], fmt = 'o')
    scatter = plt.scatter(B['MJD'],B['Mag'],label = 'B')
    
    V = results_table[results_table['Filter'] == 2]
    plt.errorbar(V['MJD'],V['Mag'], yerr = V['Error'], fmt = 'o')
    plt.scatter(V['MJD'],V['Mag'],label = 'V')
    
    gp = results_table[results_table['Filter'] == 3]
    plt.errorbar(gp['MJD'],gp['Mag'], yerr = gp['Error'], fmt = 'o')
    plt.scatter(gp['MJD'],gp['Mag'],label = 'gp')
    
    ip = results_table[results_table['Filter'] == 4]
    plt.errorbar(ip['MJD'],ip['Mag'], yerr = ip['Error'], fmt = 'o')
    plt.scatter(ip['MJD'],ip['Mag'],label = 'ip')
    
    rp = results_table[results_table['Filter'] == 5]
    plt.errorbar(rp['MJD'],rp['Mag'], yerr = rp['Error'], fmt = 'o')
    plt.scatter(rp['MJD'],rp['Mag'],label = 'rp')
    
    I = results_table[results_table['Filter'] == 6]
    plt.errorbar(I['MJD'],I['Mag'], yerr = I['Error'], fmt = 'o')
    plt.scatter(I['MJD'],I['Mag'],label = 'SDSS-I')
    
    R = results_table[results_table['Filter'] == 7]
    plt.errorbar(R['MJD'],R['Mag'], yerr = R['Error'], fmt = 'o')
    plt.scatter(R['MJD'],R['Mag'],label = 'SDSS-R')
    
    Z = results_table[results_table['Filter'] == 8]
    plt.errorbar(Z['MJD'],Z['Mag'], yerr = Z['Error'], fmt = 'o')
    plt.scatter(Z['MJD'],Z['Mag'],label = 'SDSS-Z')
    
    G = results_table[results_table['Filter'] == 9]
    plt.errorbar(G['MJD'],G['Mag'], yerr = G['Error'], fmt = 'o')
    plt.scatter(G['MJD'],G['Mag'],label = 'SDSS-G')
   
    ax = scatter.axes
    ax.invert_yaxis()
    plt.legend()
    plt.savefig('Light_curve.png')
    plt.show()

directory = input('Which directory am I looking in? ')
cutoff = input('If you are using template subtracted images, would you like to clear other data beyond a cutoff? (Optional) ')
groupid = input('Please enter the object name or groupid as in the fits header ')
sky_coord = input('Please enter the Galactic coordinates in format "RA DEC" ')
ja200_coord = input('Please enter the ja200 coordinates in "RA DEC" ')
fitting(directory,groupid,sky_coord,ja200_coord.split(),cutoff)
