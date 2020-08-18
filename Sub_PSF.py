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


plt.close('all')


class Photimage():
    
    def __init__(self,filename,groupid,sky_coord,ja200_coord):
        self.sky_coord = sky_coord
        self.ja200_coord =ja200_coord
        self.groupid = groupid
        self.hdulist=fits.open(filename)
        if filename[:2] == 'fz':
            self.fz = 1
        else:
            self.fz = 0
        self.hdr=self.hdulist[self.fz].header
        if self.hdr['GROUPID'][:9] != self.groupid:
            Exception
        self.limit=3
        self.sigma_psf=3

        self.Exposure = self.hdr['EXPTIME']
        if self.hdr['FILTER1'] == 'air':
            self.Filter =  self.hdr['FILTER']
            self.MJD = math.floor(self.hdr['MJD-OBS'])
            self.Telescope = self.hdr['TELESCOP']
            self.readnoise = self.hdr['RDNOISE']
        else:
            self.Filter =  self.hdr['FILTER1']
            self.MJD = math.floor(self.hdr['MJD'])
            self.Telescope = 'LT' 
            self.readnoise = self.hdr['READNOIS']
        self.gain = self.hdr['GAIN']
        self.wcs = WCS(self.hdr)
        self.filename=filename
        
        if self.Filter == 'B':
            self.filter_label=1
            self.Filter_simp = 'B'
        if self.Filter == 'V':
            self.filter_label=2
            self.Filter_simp = 'V'
        if self.Filter == 'gp':
            self.filter_label=3
            self.Filter_simp = 'G'
        if self.Filter == 'ip':
            self.filter_label=4
            self.Filter_simp = 'I'
        if self.Filter == 'rp':
            self.filter_label=5
            self.Filter_simp = 'R'
        if self.Filter == 'SDSS-I':
            self.filter_label=6
            self.Filter_simp = 'I'
        if self.Filter == 'SDSS-R':
            self.filter_label=7
            self.Filter_simp = 'R'
        if self.Filter == 'SDSS-Z':
            self.filter_label=8
            self.Filter_simp = 'Z'
        if self.Filter == 'SDSS-G':
            self.filter_label=9
            self.Filter_simp = 'G'
        if ('sub' in filename) == True:
            self.type = 'Subtracted'
        if ('template' in filename) == True:
            self.type = 'Template'
        else:
            self.type = 'Not'

        
    def xy(self):
        hpc_frame = wcs_to_celestial_frame(self.wcs)
        skycoord =  SkyCoord(self.sky_coord, frame=hpc_frame, unit=(u.hourangle, u.deg))
        ircs =  np.array([[skycoord.ra.deg, skycoord.dec.deg]])
        coords =  np.array(self.wcs.wcs_world2pix(ircs,1))
        y =   int((coords[0])[0])
        x =   int((coords[0])[1])
        return(x,y)
        
    def background(self,x,y):
        r=50
        x = int(x)
        y = int(y)
        data = self.hdulist[self.fz].data[x-r:x+r,y-r:y+r]
        data = np.array(data.flatten())
        data,a,b = sigmaclip(data)
        mean = np.mean(data)
        std = np.std(data)
        return(mean,std)
                   
    def Flux(self,x,y,ll,ul,r):
        x = int(x)
        y = int(y)
        data = self.hdulist[self.fz].data[x-r:x+r,y-r:y+r]
        data = (lacosmic.lacosmic(data,2,10,10, effective_gain = self.gain, readnoise = self.readnoise))[0]
        bkgrms = MADStdBackgroundRMS()
        std = bkgrms(data)
        iraffind = IRAFStarFinder(threshold=self.limit*std,
                                   fwhm=self.sigma_psf*gaussian_sigma_to_fwhm,
                                   minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
                                   sharplo=0.0, sharphi=2.0)
        daogroup = DAOGroup(2.0*self.sigma_psf*gaussian_sigma_to_fwhm)
        mmm_bkg = MMMBackground()
        psf_model = IntegratedGaussianPRF(sigma=self.sigma_psf)
        from photutils.psf import IterativelySubtractedPSFPhotometry
        photometry = IterativelySubtractedPSFPhotometry(finder=iraffind,
                                                         group_maker=daogroup,
                                                         bkg_estimator=mmm_bkg,
                                                         psf_model=psf_model,
                                                         fitter=LevMarLSQFitter(),
                                                         niters=1, fitshape=(21,21))
        

        
        result_tab = photometry(image=data)   
        
        """
        if plot == 1:
            residual_image = photometry.get_residual_image()
            print(result_tab['x_fit','y_fit'])
            plt.figure(self.filename+' data')
            plt.imshow(data, cmap='viridis',
                       aspect=1, interpolation='nearest', origin='lower')
            plt.show()
            plt.figure(self.filename+' residual')
            plt.imshow(residual_image, cmap='viridis',
                       aspect=1, interpolation='nearest', origin='lower')
            plt.show()
            plt.figure(self.filename+' PSF')
            plt.imshow(data-residual_image, cmap='viridis',
                       aspect=1, interpolation='nearest', origin='lower')
            plt.show()
        """
        
        if len(result_tab) > 5:
            return(0,0) 
        if len(result_tab) ==0:
            print('None')
            return(0,0) 
        result_tab['Minus'] = np.zeros(len(result_tab))
        for i in range(len(result_tab)):
            #if 18.5 < result_tab['x_fit'][i] < 28.5 and 18.5 < result_tab['y_fit'][i] < 28.5:
            if ll < result_tab['x_fit'][i] < ul and ll < result_tab['y_fit'][i] < ul:
                result_tab['Minus'][i] = 1
            else:
                result_tab['Minus'][i] = 0
        mask = result_tab['Minus'] == 1.0
        result_tab = result_tab[mask]
        if len(result_tab) != 1:
            return(0,0)   
        flux_counts = float(result_tab['flux_fit'][0])
        flux_unc = float(result_tab['flux_unc'][0])
        flux_unc = flux_unc/flux_counts
        error = np.sqrt((flux_counts/self.gain) + (self.sigma_psf*gaussian_sigma_to_fwhm**2*np.pi*self.readnoise))
        return(flux_counts,flux_unc)

        
    def Mags(self):

        if self.Filter == 'B' or self.Filter == 'V':
            v = Vizier(columns=['RAJ2000', 'DEJ2000','B-V','Bmag','Vmag'])
            catalog = 'II/336'
        else:
            v = Vizier(columns=['RAJ2000', 'DEJ2000','gmag','rmag','imag','zmag'])
            catalog = 'II/349'
     
        result = v.query_region(coord.SkyCoord(ra=self.ja200_coord[0], dec=self.ja200_coord[1],unit=(u.deg, u.deg),frame='icrs'), radius=3*u.arcmin, catalog = catalog)
        for table_name in result.keys():        
            table = result[table_name]
            print(table.colnames)

            if self.Filter == 'B' or self.Filter == 'V':
                df2 = pd.DataFrame(np.zeros([len(table),5]),columns=['x', 'y', 'B-V', 'B', 'V'])
            else:
                df2 = pd.DataFrame(np.zeros([len(table),6]),columns=['x', 'y', 'R', 'Z', 'I', 'G'])

            for i in range(len(table)):
                ra =  table['RAJ2000'][i]
                de =  table['DEJ2000'][i]
                pixa,pixb = self.wcs.wcs_world2pix(ra,de,1)
                
                if self.Filter == 'B' or self.Filter == 'V':
                    new_1 = pd.DataFrame({'x':[pixa]},index = [i])
                    new_2 = pd.DataFrame({'y':[pixb]},index = [i])
                    new_3 = pd.DataFrame({'B-V':[table['B-V'][i]]},index = [i])
                    new_4 = pd.DataFrame({'B':[table['Bmag'][i]]},index = [i])#B-V*(colour_term)
                    new_5 = pd.DataFrame({'V':[table['Vmag'][i]]},index = [i])
                    df2.update(new_1)
                    df2.update(new_2)
                    df2.update(new_3)
                    df2.update(new_4)
                    df2.update(new_5)
                
                else:
                    new_1 = pd.DataFrame({'x':[pixa]},index = [i])
                    new_2 = pd.DataFrame({'y':[pixb]},index = [i])
                    new_3 = pd.DataFrame({'R':[table['rmag'][i]]},index = [i])#r-i*(colour_term)
                    new_4 = pd.DataFrame({'Z':[table['zmag'][i]]},index = [i])#i-z
                    new_5 = pd.DataFrame({'I':[table['imag'][i]]},index = [i])#i-z*(colour_term)
                    new_6 = pd.DataFrame({'G':[table['gmag'][i]]},index = [i])#g-r
                    df2.update(new_1)
                    df2.update(new_2)
                    df2.update(new_3)
                    df2.update(new_4)
                    df2.update(new_5)
                    df2.update(new_6)
        return(df2)    
        

def ZP(ob):
    colour_terms = pd.read_csv('Telescopes.csv')
    df2 = ob.Mags()
    zpsum = 0
    n = 0
    for i in range(len(df2)):
        x = df2['x'][i]
        y = df2['y'][i]    

        count,flux_unc = ob.Flux(y,x,18.5,28.5,25)
        if count == 0:
            continue
        else:
            pass
        if isinstance(df2[ob.Filter_simp][i], float) == True:
            pass
        else:
            continue
        if ob.Telescope == '1m0-06':
            ob.Telescope = '1m0-05'
        colour = np.float(colour_terms.loc[colour_terms['Telescope'] == ob.Telescope, ob.Filter_simp].iloc[0])
        measure = -2.5*np.log10(count/ob.Exposure)
        if ob.Filter_simp == 'B' or ob.Filter_simp == 'V':
            colour_2 = df2['B-V'][i]
        if ob.Filter_simp == 'Z': 
            if isinstance(df2['I'][i], float) == True:
                colour_2 = df2['Z'][i] - df2['I'][i]
            else:
                continue
        if ob.Filter_simp == 'I':
            if isinstance(df2['Z'][i], float) == True:
                colour_2 = df2['Z'][i] - df2['I'][i]
            else:
                continue
        if ob.Filter_simp == 'R':
            if isinstance(df2['I'][i], float) == True:
                colour_2 = df2['R'][i] - df2['I'][i]
            else:
                continue
        if ob.Filter_simp == 'G':
            if isinstance(df2['R'][i], float) == True:
                colour_2 = df2['G'][i] - df2['R'][i]
            else:
                continue           
        n = n+1
        zp = df2[ob.Filter_simp][i] - measure - colour*colour_2    
        if isinstance(zp, float) == True:
            pass
        else:
            continue
        zpsum = zpsum + zp 
    if n == 0:
        return(0)
    else:
        return(zpsum/n)

 
def Return(filename,groupid,sky_coord,ja200_coord):
    temp = 0
    ob = Photimage(filename,groupid,sky_coord,ja200_coord) 
    if ob.type == 'Template': 
        temp=1
    zp= ZP(ob)
    if zp == 0:
        return(0,0,0,0,0,0,0)
    x,y = ob.xy()

    if temp == 1: 
        ul =29
        ll=15
        r = 20
    if temp == 0:
        r=25
        ll=18.5
        ul=28.5
        
    counts,err = ob.Flux(x,y,ll,ul,r)
  
    if counts == 0:
        ob.limit = ob.limit - 0.5
        ob.sigma_psf = ob.sigma_psf+ 0.5
        counts,err = ob.Flux(x,y,ll,ul,r)
        if counts == 0:
            ob.limit = ob.limit 
            ob.sigma_psf = ob.sigma_psf - 0.5
            counts,err = ob.Flux(x,y,ll,ul,r)
            if counts == 0:
                return(0,0,0,0,0,0,0)
        


    mean,std=ob.background(x,y)    
    npix = (gaussian_sigma_to_fwhm*ob.sigma_psf)**2*np.pi
    sky = mean*npix
    term1=(counts+sky)/ob.gain
    term2=npix*(ob.readnoise+std)
    skyerror = (np.sqrt(term1+term2))
    err = np.sqrt(skyerror**2+err**2)
    filter_label = ob.filter_label
    return(ob.MJD,zp,counts,ob.Exposure,err,filter_label,temp)


#print(Return('/Users/eleonoraparrag/Documents/Python/LCO_combine/ORIG/sub_gp_0708.fits', 'SN2019hcc', '21:00:20.930 -21:20:36.06', '315.08720833 -21.34335',58670))