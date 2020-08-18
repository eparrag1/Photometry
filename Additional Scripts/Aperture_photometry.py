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


plt.close('all')
#sky_coord = '02:26:18.55 -09:50:09.0'
sky_coord = '21:00:20.930 -21:20:36.06'
#ja200_coord = (36.57729167, -9.83583333)
ja200_coord =(315.08720833,-21.34335)


colour_terms = pd.DataFrame(np.zeros([9,7]),columns=['Telescope','B', 'V', 'R', 'Z', 'I', 'G'])
new_0 = pd.DataFrame({'Telescope':['1m0-03'],'B':[-0.025],'V':[0.017],'R':[-0.005],'Z':['--'],'I':[0.007],'G':[0.137]}, index = [8])
new_1 = pd.DataFrame({'Telescope':['1m0-04'],'B':[-0.024],'V':[-0.014],'R':[0.027],'Z':['--'],'I':[0.036],'G':[0.109]}, index = [0])
new_2 = pd.DataFrame({'Telescope':['1m0-05'],'B':[-0.035],'V':[0],'R':[-0.002],'Z':['--'],'I':[0.019],'G':[0.120]}, index = [1])
new_3 = pd.DataFrame({'Telescope':['1m0-08'],'B':[-0.039],'V':[-0.005],'R':[-0.004],'Z':['--'],'I':[0.024],'G':[0.114]}, index = [2])
new_4 = pd.DataFrame({'Telescope':['1m0-10'],'B':[-0.030],'V':[-0.019],'R':[-0.001],'Z':['--'],'I':[0.013],'G':[0.112]}, index = [3])
new_5 = pd.DataFrame({'Telescope':['1m0-11'],'B':[-0.025],'V':[0.017],'R':[-0.005],'Z':['--'],'I':[0.007],'G':[0.137]}, index = [4])
new_6 = pd.DataFrame({'Telescope':['1m0-12'],'B':[-0.030],'V':[-0.019],'R':[-0.001],'Z':['--'],'I':[0.013],'G':[0.112]}, index = [5])
new_7 = pd.DataFrame({'Telescope':['1m0-13'],'B':[-0.030],'V':[-0.019],'R':[-0.001],'Z':['--'],'I':[0.013],'G':[0.112]}, index = [6])
new_8 = pd.DataFrame({'Telescope':['LT'],'B':['--'],'V':['--'],'R':[0.024],'Z':[0.237],'I':[0],'G':[0.057]}, index = [7])
colour_terms.update(new_0)
colour_terms.update(new_1)
colour_terms.update(new_2)
colour_terms.update(new_3)
colour_terms.update(new_4)
colour_terms.update(new_5)
colour_terms.update(new_6)
colour_terms.update(new_7)
colour_terms.update(new_8)
 
def properties(hdulist,a):
    origin = hdulist[a].header['ORIGIN']
    exposure = hdulist[a].header['EXPTIME']
    gain = hdulist[a].header['GAIN']
    airmass = hdulist[a].header['AIRMASS']
    #Different telescopes have different header labels
    if hdulist[a].header['FILTER1'] == 'air':
        MJD = hdulist[a].header['MJD-OBS']
        filter_ =  hdulist[a].header['FILTER']
        sn = 'Yes'
        telescope = hdulist[a].header['TELESCOP']
        sigma = hdulist[a].header['L1SIGMA']
        readnoise = hdulist[a].header['RDNOISE']
    else:
        MJD = hdulist[0].header['MJD']
        sigma = hdulist[0].header['STDDEV']
        readnoise = hdulist[a].header['READNOIS']
        telescope = 'LT'
        filter_ =  hdulist[0].header['FILTER1']
        if hdulist[0].header['GROUPID'][:9] == 'SN2019hcc':
            sn = 'Yes'
        if hdulist[0].header['GROUPID'][:9] != 'SN2019hcc':
            sn = 'No'
            #Different LCO telescopes have different zeropoints 
    return(readnoise,exposure,gain,airmass,filter_,MJD,sn,telescope)
                
#####GAUSSIAN FITTING
def func(x, amp, cen, wid):
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

def threed_func(x, y, amp, cenx, ceny, wid):
    return amp * np.exp(-((x-cenx)**2+(y-ceny)**2) / (2*wid**2))

def threed_array(data,w,x,y):
    k = w
    r = 2*k
    x3d = np.linspace(0,r-1,r)
    y3d = np.zeros(r)
    
    for i in range(r-1):
        newy = i*np.ones(r)   
        newx = np.linspace(0,r-1,r)
        y3d = np.append(y3d,newy)
        x3d = np.append(x3d,newx)
    

    data = data[x-k:x+k,y-k:y+k]
    z3d = data.flatten()
    return(x3d,y3d,z3d)
    

def skycoord(hdulist,a):
    hdr = hdulist[a].header
    wcs = WCS(hdr)
    hpc_frame = wcs_to_celestial_frame(wcs)
    skycoord =  SkyCoord(sky_coord, frame=hpc_frame, unit=(u.hourangle, u.deg))
    ircs =  np.array([[skycoord.ra.deg, skycoord.dec.deg]])
    coords =  np.array(wcs.wcs_world2pix(ircs,1))
    y =   int((coords[0])[0])
    x =   int((coords[0])[1])
    return(x,y,wcs)
                
def reverse_starxy(wcs,x,y):
    world = wcs.wcs_world2pix(x,y,1)
    x,y = world
    return(x,y)
    

def Mags(x,y,wcs):    
    v = Vizier(columns=['RAJ2000', 'DEJ2000','gmag','rmag','imag','zmag'])
    result = v.query_region(coord.SkyCoord(ra=x, dec=y,unit=(u.deg, u.deg),frame='icrs'), radius=3*u.arcmin, catalog = 'II/349')
    for table_name in result.keys():        
        table = result[table_name]
        df2 = pd.DataFrame(np.zeros([len(table),6]),columns=['x', 'y', 'R', 'Z', 'I', 'G'])
        for i in range(len(table)):
            ra =  table['RAJ2000'][i]
            de =  table['DEJ2000'][i]
            pixb,pixa = reverse_starxy(wcs,ra,de)
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
    
def MagsBV(x,y,wcs):   
    v = Vizier(columns=['RAJ2000', 'DEJ2000','B-V','Bmag','Vmag'])
    result = v.query_region(coord.SkyCoord(ra=x, dec=y,unit=(u.deg, u.deg),frame='icrs'), radius=5*u.arcmin, catalog = 'II/336')
    for table_name in result.keys():        
        table = result[table_name]
        df2 = pd.DataFrame(np.zeros([len(table),5]),columns=['x', 'y', 'B-V', 'B', 'V'])
        for i in range(len(table)):
            ra =  table['RAJ2000'][i]
            de =  table['DEJ2000'][i]
            pixb,pixa = reverse_starxy(wcs,ra,de)
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
    return(df2)    
    
def parameters(data,x,y):
    r = 100
    x = int(x)
    y = int(y)
    #setting up x and y for Gaussian fitting (centering y on zero by minusing mean)
    xpeak = np.linspace(1,r,num = r)  
    ysd = data[int(x-r/2):int(x+r/2),y]
    new1 = data[x-50:x-15,y] 
    new2 = data[x+15:x+50,y]
    new3 = np.concatenate((new1,new2))
    mean = np.sum(new3)/70
    ypeak = data[int(x-r/2):int(x+r/2),y] - mean
                
    #Finding standard deviation of y
    sd1 = 0
    for i in range(70):
        sd1 = sd1 + (new3[i] - mean)**2
        sd = np.sqrt(sd1/r)  
                
    #Guessing initial parameters for Gaussian
    a = ypeak.max()
    b = r/2
    c = 10
    
    return(sd,a,b,c,xpeak,ypeak)
                


def gaussian(filename,xpeak,ypeak,plot,sd,a,b,c,lim,filter_):
    
    def minfunc(params):
        return sum((ypeak-func(xpeak,params[0],params[1],params[2]))**2)
    if sd > lim:
        popt = optimize.fmin(minfunc,(a,b,c))
        gaussy = func(xpeak, *popt)
        stddev = gaussy.max()/sd  
        if stddev > lim:    
            if plot == 'Gauss':
                plt.figure(filename)
                plt.scatter(xpeak, ypeak, label = stddev)
                plt.axhline(sd, color = 'green', label = '1 sigma')
                plt.axhline(3*sd, color = 'red', label = '3 sigma')
                plt.plot(xpeak, func(xpeak, *popt), label = filter_)
                plt.legend()
                plt.show()
            gaussy = func(xpeak, *popt)
            FWHM = abs(2*np.sqrt(2*np.log(2))*popt[2])
            return(FWHM,popt)
        else:
            return(0,(0,0,0))
    else:
        return(0,(0,0,0))
 
def integral(x,y,a,b,c,d,r,FWHM):
    f = lambda y, x: threed_func(x,y,a,b,c,d)
    intlim = int(math.ceil(3*FWHM))
    x1 =r/2-intlim
    x2 =r/2+intlim
    y1 =r/2-intlim
    y2 =r/2+intlim
    integral = integrate.dblquad(f, x1, x2, lambda x: y1, lambda x: y2)
    return(integral)                       

def aperture_count(x,y,data, FWHM, gain, readnoise):
    count2 = 0
    count3 = 0
    count4 = 0
    point1 = 0
    point = 0
    sigma_clipped = np.array([])
    x = int(x)
    y = int(y)
    r = int(math.ceil(5*FWHM))
    if r > 50:
        r = 50
    data = data[x-r:x+r,y-r:y+r]
    data = (lacosmic.lacosmic(data,2,10,10, effective_gain = gain, readnoise = readnoise))[0]
    x = r
    y = r
    #Counts within annulus around SN and find mean
    for i in range(x-r,x+r):
        for j in range(y-r,y+r):
            if (3*FWHM)**2 < (i-x)**2+(j-y)**2 < (5*FWHM)**2:
                count3  = count3 + data[i,j] 
                sigma_clipped = np.append(sigma_clipped, data[i,j])
                point = point+1
    sigma_clipped = sigma_clip(sigma_clipped, sigma=2)
    mean = np.mean(sigma_clipped)                        
    #mean =  count3/point
                        
    #Counts with 3FWHM radius around SN, each point minus mean from annulus
    for i in range(x-r,x+r):
        for j in range(y-r,y+r):
            if (i-x)**2+(j-y)**2 < (3*FWHM)**2:
                count4  = count4 + data[i,j] 
                point1 = point1 + 1 
    count2 = count4 - mean*point1  
    
    if count2 < 0:
        return [0]
    else:
        return [count2,count3,count4,point1]

    

def appender(array,MJD,count2,exposure,airmass,skyerror,point1,count3,count4,gain,filter_,zp):
    i = 0
    while i < len(array):
        if math.floor(array[i,0]) == math.floor(MJD):
            array[i,1] = array[i,1] + count2
            array[i,2] = array[i,2] + exposure
            array[i,3] = array[i,3] + airmass
            array[i,4] = array[i,4] + 1
            array[i,6] = array[i,6] + zp
            array[i,7] = array[i,7] + (skyerror)**2
            array[i,8] = array[i,8] + count3
            array[i,9] = array[i,9] + count4
            break
        i = i+1
    else:
        array = np.vstack([array,[math.floor(MJD), count2, exposure, airmass, 1, 1, zp, (skyerror)**2, count3, count4, filter_, 1, gain]])
    return(array)
    

def magnitude(zp,MJD,counts,exposure,airmass,extinction):
    mag = zp  - 2.5*np.log10(counts/exposure)
    return(mag)
   
def sky_error(FWHM,a,b,c,data,wcs,gain, readnoise, filter_):  
    spotx,spoty =  reverse_starxy(wcs,315.10533939,-21.32286863)
    #spotx,spoty = reverse_starxy(wcs, 36.55986926,-9.8808634)
    r = 100
    x3d = np.linspace(0,r-1,r)
    y3d = np.zeros(r)
    
    for i in range(r-1):
        newy = i*np.ones(r)   
        newx = np.linspace(0,r-1,r)
        y3d = np.append(y3d,newy)
        x3d = np.append(x3d,newx)
    
    z =  threed_func(x3d, y3d, a, b, b, c)
    newx,newy,newz = threed_array(data,50,int(spotx),int(spoty))
    combz = z+newz

    counts =  integral(x3d,y3d,a,b,b,c,r,FWHM)
    integral_counts = counts[0]

    combz = combz.reshape(100,100)
    sd,a,b,c,xpeak,ypeak = parameters(combz,50,50)
    
    FWHM,(a,b,c) =  gaussian('Name',xpeak,ypeak,'None',sd,a,b,c,0, filter_)
    aperture = aperture_count(50,50,combz, FWHM, gain, readnoise)[0]
    skyerror = np.abs(aperture-integral_counts)/aperture
    return(skyerror)
    
def error(skyerror):
    err = np.sqrt(((2.5/np.log(10)) * np.sqrt(skyerror))**2 + 0.03**2 + 0.011**2)
    return(err)


         
def star_magnitude(filter_,hdulist,fz,wcs, which, telescope):
    if which == 1: #IGRZ
        df2 = Mags(ja200_coord[0],ja200_coord[1],wcs)
    if which ==2: #BV
        df2 = MagsBV(ja200_coord[0],ja200_coord[1],wcs)
    zpsum = 0
    n = 0
    for i in range(len(df2)):
        x = df2['x'][i]
        y = df2['y'][i]    
        prop = properties(hdulist,fz)
        exposure = prop[2]
        readnoise = prop[1]
        gain = prop[3]
        params = parameters(hdulist[fz].data,x,y)
        sd,a,b,c,xpeak,ypeak = params
        #if df2[filter_][i] == 0.0 or df2[filter_][i] == '--':
        #    continue
        #    print 'NO'
        if isinstance(df2[filter_][i], float) == True:
            pass
        else:
            continue
        FWHM,(a,b,c) = gaussian('None',xpeak,ypeak,'None',sd,a,b,c,5,filter_)
        if FWHM == 0:
            continue
                
        else:
            pass 
        apertures = aperture_count(x,y,hdulist[fz].data,FWHM, gain, readnoise)
        if apertures == 0:
            continue
                
        else:
            pass
        count2 = apertures[0]
        if telescope == '1m0-06':
            telescope = '1m0-05'
        colour = colour_terms.loc[colour_terms['Telescope'] == telescope, filter_].iloc[0]
        measure = -2.5*np.log10(count2/exposure)
        n = n+1
        if which == 2:
            colour_2 = df2['B-V'][i]
        if (filter_ == 'Z' or filter_ == 'I'):
            colour_2 = df2['Z'][i] - df2['I'][i]
        if filter_ == 'R':
            colour_2 = df2['R'][i] - df2['I'][i]
        if filter_ == 'G':
            colour_2 = df2['G'][i] - df2['R'][i]
        zp = df2[filter_][i] - measure - colour*colour_2
        zpsum = zpsum + zp  
    if n == 0:
        return(0)
    else:
        return(zpsum/n)

        
 
def fitting(directory):
    #setting up empty arrays to hold the data for each filter
    lightB = np.zeros((1,13))
    lightV = np.zeros((1,13))
    lightI = np.zeros((1,13))
    lightR = np.zeros((1,13))
    lightZ = np.zeros((1,13))
    lightG = np.zeros((1,13))
    lightgp = np.zeros((1,13))
    lightip = np.zeros((1,13))
    lightrp = np.zeros((1,13))
    
    for filename in os.listdir(directory):
        if filename != '.DS_Store':
            print(filename)
            hdulist = fits.open(directory + filename)
            #fz files have data under different index
            if filename[:2] == 'fz':
                fz = 1
            else:
                fz = 0
            x,y,wcs = skycoord(hdulist,fz)
            readnoise,exposure,gain,airmass,filter_,MJD,sn,telescope = properties(hdulist,fz)
            print('FILTER', filter_)
            data =  hdulist[fz].data
            
            if sn == 'No':
                continue
            else:
                pass
            sd,a,b,c,xpeak,ypeak = parameters(data,x,y)
            #Note - plotting not recommended!
            FWHM,(a,b,c) = gaussian(filename,xpeak,ypeak,'None',sd,a,b,c,0,filter_)
            if FWHM == 0:
                continue
            else:
                pass
            aperture = aperture_count(x,y,data,FWHM, gain, readnoise)
            if aperture[0] == 0:
                continue
            else:
                pass
            count2,count3,count4,point1 = aperture
            
            skyerror = sky_error(FWHM,a,b,c,data,wcs,gain, readnoise, filter_)
            if skyerror > 10:
                continue
            else:
                pass

            if filter_ == 'ip' or filter_ == 'SDSS-I':
                zp = star_magnitude('I',hdulist,fz,wcs,1,telescope)
            if filter_ == 'gp' or filter_ == 'SDSS-G':
                zp = star_magnitude('G',hdulist,fz,wcs,1,telescope)
            if filter_ == 'rp' or filter_ == 'SDSS-R':
                zp = star_magnitude('R',hdulist,fz,wcs,1,telescope)
            if filter_ == 'SDSS-Z':
                zp = star_magnitude('Z',hdulist,fz,wcs,1,telescope)
            if filter_ == 'B':
                zp = star_magnitude('B',hdulist,fz,wcs,2,telescope)
            if filter_ == 'V':
                zp = star_magnitude('V',hdulist,fz,wcs,2,telescope)
                
            if zp == 0:
                continue
            else:
                pass
            
            #append info to arrays

            print(filter_)
            

            if filter_ == 'B':
                lightB = appender(lightB,MJD,count2,exposure,airmass,skyerror,point1,count3,count4,gain,1,zp)
                lightB[:,11] = 0.23
            if filter_ == 'V':
                lightV = appender(lightV,MJD,count2,exposure,airmass,skyerror,point1,count3,count4,gain,2,zp)
                lightV[:,11] = 0.12
            if filter_ == 'gp':
                lightgp = appender(lightgp,MJD,count2,exposure,airmass,skyerror,point1,count3,count4,gain,3,zp)
                lightgp[:,11] = 0.14
            if filter_ == 'ip':
                lightip = appender(lightip,MJD,count2,exposure,airmass,skyerror,point1,count3,count4,gain,4,zp)
                lightip[:,11] = 0.06
            if filter_ == 'rp':
                lightrp = appender(lightrp,MJD,count2,exposure,airmass,skyerror,point1,count3,count4,gain,5,zp)
                lightrp[:,11] = 0.08
            if filter_ == 'SDSS-I':
                lightI = appender(lightI,MJD,count2,exposure,airmass,skyerror,point1,count3,count4,gain,6,zp)
                lightI[:,11] = 0.05
            if filter_ == 'SDSS-R':
                lightR = appender(lightR,MJD,count2,exposure,airmass,skyerror,point1,count3,count4,gain,7,zp)
                lightR[:,11] = 0.1
            if filter_ == 'SDSS-Z':
                lightZ = appender(lightZ,MJD,count2,exposure,airmass,skyerror,point1,count3,count4,gain,8,zp)
                lightZ[:,11]  = 0.04
            if filter_ == 'SDSS-G':
                lightG = appender(lightG,MJD,count2,exposure,airmass,sigma,skyerror,point1,count3,count4,gain,9,zp)
                lightG[:,11]  = 0.164
    
    return(lightB,lightV,lightgp,lightip,lightrp,lightI,lightZ,lightG,lightR)
    

def arrays(directory):
    arrays = fitting(directory)
    plt.figure("Light Curves")
    plt.xlabel('Modified Julian Date')
    plt.ylabel('Instrumental Magnitude')
    for i in range(len(arrays)):
        array = arrays[i]
        array = np.delete(array,0,0)
        
        #LABELS
        if array[1,10] == 1.0:
            filter_label = 'B'
        if array[1,10] == 2.0:
            filter_label = 'V'
        if array[1,10] == 3.0:
            filter_label = 'gp'
        if array[1,10] == 4.0:
            filter_label = 'ip'
        if array[1,10] == 5.0:
            filter_label = 'rp'
        if array[1,10] == 6.0:
            filter_label = 'SDSS-I'
        if array[1,10] == 7.0:
            filter_label = 'SDSS-R'
        if array[1,10] == 8.0:
            filter_label = 'SDSS-Z'
        if array[1,10] == 9.0:
            filter_label = 'SDSS-G' 
            
        #CALCULATION
        array[:,5] = error(array[:,7]/array[:,4])
        print(array[:,5])
        array[:,1] = magnitude((array[:,6]/array[:,4]),array[:,0],array[:,1],array[:,2],(array[:,3]/array[:,4]),array[:,11])

        #PLOTTING
        array[:,0], array[:,1] = zip(*sorted(zip(array[:,0], array[:,1])))
        scatter = plt.scatter(array[:,0], array[:,1],label = filter_label)
        plt.errorbar(array[:,0], array[:,1], yerr = array[:,5])
        

    plt.legend()
    ax = scatter.axes
    ax.invert_yaxis()
    plt.show()
    return(arrays)
      
def other(directory):

    for filename in os.listdir(directory):
        if filename != '.DS_Store':
            print(filename)
            hdulist = fits.open(directory + filename)
            #fz files have data under different index
            if filename[:2] == 'fz':
                fz = 1
            else:
                fz = 0
            x,y,wcs = skycoord(hdulist,fz)
            readnoise,exposure,gain,airmass,filter_,MJD,sn,telescope = properties(hdulist,fz)
            print('FILTER', filter_)
            data =  hdulist[fz].data
            
            if sn == 'No':
                continue
            else:
                pass
            sd,a,b,c,xpeak,ypeak = parameters(data,x,y)
            #Note - plotting not recommended!
            FWHM,(a,b,c) = gaussian(filename,xpeak,ypeak,'Gauss',sd,a,b,c,0,filter_)
            if FWHM == 0:
                continue
            else:
                pass
            aperture = aperture_count(x,y,data,FWHM, gain, readnoise)
            if aperture[0] == 0:
                continue
            else:
                pass
            count2,count3,count4,point1 = aperture
            print(count2,count3,count4,point1)
            
#arrays('/Users/eleonoraparrag/documents/Python/Other_SN/SN_other/') 
#arrays('/Users/eleonoraparrag/documents/Python/SN_Master/') 
#arrays('/Users/eleonoraparrag/documents/Python/LCO_combine/Weird/') 
#Get I/G/R/Z only for now      
other('/Users/eleonoraparrag/documents/Python/LCO_combine/Weird/')

       

"""
def Line_Profile():              
    lp = LineProfiler()
    lp.add_function(gaussian)   # add additional function to profile
    lp.add_function(Mags)   # add additional function to profile
    lp.add_function(star_magnitude)   # add additional function to profile
    lp_wrapper = lp(fitting)
    lp_wrapper('/Users/eleonoraparrag/documents/Python/SN_IRG_small/')
    lp.print_stats()
 
Line_Profile()
"""
