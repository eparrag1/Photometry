import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import optimize
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.patches as mpatches
from scipy import interpolate

directory = '/Users/eleonoraparrag/Documents/Spectroscopy/'
plt.close('all')
def flux(mags):
    
    def bbody(lam,T,A):
        Blam = A*(2*6.626e-27*(3e10)**2/(lam*1e-8)**5)/(np.exp(6.627e-27*3e10/(lam*1e-8*1.38e-16*T))-1)
        return(Blam)


    # effective wavelengths of filters:
    wle = {'S':2030,'D':2231,'P':2634,'u': 3560 , 'g': 4830 , 'r':6260 , 'i': 7670 , 'z': 9100 
           , 'U': 3600 , 'B': 4380 , 'V': 5450 , 'R': 6410 , 'I': 7980, 'J':12200, 'H':16300,
           'K': 21900}
     
     
    # Reference fluxes for converting magnitudes to flux
    zp = {'S': 536.2, 'D': 463.7, 'P': 412.3 ,'u': 859.5 , 'g': 466.9 , 'r': 278.0 , 'i': 185.2 
          , 'z': 131.5 , 'U': 417.5 , 'B': 632 , 'V': 363.1 , 'R': 217.7 , 'I': 112.6, 'J':31.47
          , 'H':11.38, 'K':3.961}
    
    wl = []
    fref = []
    
    #mags = np.array([20.26,18.64,18.79,18.41]) 
    
    bands = 'griz'
    #bands = 'BVRI'
    
    for i in bands:
        wl.append(wle[i])
        fref.append(zp[i]*1e-11) # because reference fluxes in units of 10^-11 erg/s/cm2/A
     
    wl = np.array(wl)
    fref = np.array(fref)
    
    
    # initialize constants
    H0 = 72.                         # Hubble constant
    WM = 0.27                        # Omega(matter)
    WV = 1.0 - WM - 0.4165/(H0*H0)  # Omega(vacuum) or lambda
    
    WR = 0.        # Omega(radiation)
    WK = 0.        # Omega curvaturve = 1-Omega(total)
    c = 299792.458 # velocity of light in km/sec
    Tyr = 977.8    # coefficent for converting 1/H into Gyr
    DTT = 0.0      # time from z to now in units of 1/H0
    DCMR = 0.0     # comoving radial distance in units of c/H0
    DA = 0.0       # angular size distance
    DL = 0.0       # luminosity distance
    DL_Mpc = 0.0
    a = 1.0        # 1/(1+z), the scale factor of the Universe
    az = 0.5       # 1/(1+z(object))
    
    h = H0/100.
    WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
    WK = 1-WM-WR-WV
    z  = 0.044
    az = 1.0/(1+1.0*z)
    n=1000         # number of points in integrals
    
    
    for i in range(n):
        a = az+(1-az)*(i+0.5)/n
        adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        DTT = DTT + 1./adot
        DCMR = DCMR + 1./(a*adot)
    
    DTT = (1.-az)*DTT/n
    DCMR = (1.-az)*DCMR/n
    
    ratio = 1.00
    x = np.sqrt(abs(WK))*DCMR
    if x > 0.1:
        if WK > 0:
            ratio =  0.5*(np.exp(x)-np.exp(-x))/x
        else:
            ratio = np.sin(x)/x
    else:
        y = x*x
        if WK < 0: y = -y
        ratio = 1. + y/6. + y*y/120.
    
    DCMT = ratio*DCMR
    DA = az*DCMT
    
    DL = DA/(az*az)
    
    DL_Mpc = (c/H0)*DL
    
    #############################################
    
    SN_distance = DL_Mpc*6.009974e+26 # convert Mpc to cm, since flux in erg/s/cm2/A
    
    # convert mags to flux (and plot):
    cons = 4*np.pi*SN_distance**2*fref
    flux = cons*10**(-0.4*mags)
    wl = np.array(wl)
    flux = np.array(flux/1e43)



    return(wl,flux)

def spectra_plot(night,add,adj,plot):
    for filename in os.listdir(directory + night):
        if filename[-6:] == 'e.asci':
            print(filename)
            data = np.genfromtxt(directory+night+'/'+filename, dtype=None)
            x =  data[:,0]
            y =  data[:,1]
            if adj == 'shift':
                y = y*1e16
            print(x.shape)
            print(y.shape)
            if plot == 'yes':
                plt.plot(x,y+add)
                plt.show()
    return(x,y)


def blackbody(wav, T, b):
    wav = wav*1e-10
    h = 6.63e-34
    c = 3e8
    kb = 1.38e-23
    frac1 = (2*h*c**2)/wav**5
    frac2 = np.exp((h*c)/(wav*kb*T)) - 1
    wav = wav*1e9
    bb = (frac1/frac2)/b    
    return(bb)
   

def blackbody_fit(x,y,add,plot):
    guess = 7000
    b = 7e8
    print(x.shape)
    print(y.shape)
    def minfunc(A):
        T,b = A
        return sum((y-blackbody(x,T,b))**2)
    
    T,b = optimize.fmin(minfunc,(guess,b))
    bb = blackbody(x,T, b)
    
    if plot == 'yes':
        plt.figure()
        plt.plot(x,y)
        plt.plot(x,bb+add, label = T)
        plt.legend()
        plt.show()
    return(T,b)

def blackbody_curve_fit(x,y,add,plot):
    guess = 7000
    b = 7e8
    print(x.shape)
    print(y.shape)
    
    popt,cov  = curve_fit(blackbody,x,y,p0=(7000,7e8))
    print(popt)
    print(cov)
    bb = blackbody(x,*popt)
    T,b = popt
    if plot == 'yes':
        plt.figure()
        plt.plot(x,y)
        plt.plot(x,bb+add, label = T)
        plt.legend()
        plt.show()
    return(T,b)
    
def moving_average(N,y):
    n = np.copy(y)
    for i in range(N):
        for i in range(1,len(n)-1):
            n[i] = (n[i-1]+n[i+1])/2
    return(n)

def mangle(x,y,mags,plot):
    wl,fref = flux(mags)
    print(wl,fref)
    #T,b =  blackbody_fit(wl,fref,0,'no')
    T,b=  blackbody_curve_fit(wl,fref,0,'no')

    bb = blackbody(x,T,b)

    
    a = fref.argmax()
    wl_max = (2.838e-3/wl[a])*1e10
    print(wl_max)

    #y = bb + y_new
    y = bb*y
    y = y*1e16
    print('BLACKBODY TEMP ',T)
    
    if plot == 'yes':
        plt.plot(x,y,zorder=50)   
        plt.scatter(wl,fref)
        plt.plot(x,bb)
        plt.show()
    
    return(x,y,T)
    
def cardelli(lamb, Av, Rv=3.1, Alambda = True):
    """
    Cardelli extinction Law
    input:
        lamb    <float>    wavelength of the extinction point !! in microns !!
    output:
        tau        <float> returns tau as in redflux = flux*exp(-tau)
    keywords:
        Alambda        <bool>  returns +2.5*1./log(10.)*tau
        Av        <float>    extinction value (def: 1.0)
        Rv        <float> extinction param. (def: 3.1)
    """

    if type(lamb) == float:
        _lamb = np.asarray([lamb])
    else:
        _lamb = lamb[:]

    #init variables
    x = 10000./(_lamb) #wavenumber in um^-1
    a = np.zeros(np.size(x))
    b = np.zeros(np.size(x))
    #Infrared (Eq 2a,2b)
    ind = np.where ((x >= 0.3) & (x < 1.1))
    a[ind] =  0.574*x[ind]**1.61
    b[ind] = -0.527*x[ind]**1.61
    #Optical & Near IR
    #Eq 3a, 3b
    ind = np.where ((x >= 1.1) & (x <= 3.3))
    y = x[ind]-1.82
    a[ind] = 1. + 0.17699*y   - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b[ind] =      1.41338*y   + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    #UV
    #Eq 4a, 4b
    ind = np.where ((x >= 3.3) & (x <= 8.0))
    a[ind] =  1.752 - 0.316*x[ind] - 0.104/((x[ind]-4.67)**2+0.341)
    b[ind] = -3.090 + 1.825*x[ind] + 1.206/((x[ind]-4.62)**2+0.263)

    ind = np.where ((x >= 5.9) & (x <= 8.0))
    Fa     = -0.04473*(x[ind]-5.9)**2 - 0.009779*(x[ind]-5.9)**3
    Fb     =  0.21300*(x[ind]-5.9)**2 + 0.120700*(x[ind]-5.9)**3
    a[ind] = a[ind] + Fa
    b[ind] = b[ind] + Fb
    #Far UV
    #Eq 5a, 5b
    ind = np.where ((x >= 8.0) & (x <= 10.0))
    #Fa = Fb = 0
    a[ind] = -1.073 - 0.628*(x[ind]-8.) + 0.137*((x[ind]-8.)**2) - 0.070*(x[ind]-8.)**3
    b[ind] = 13.670 + 4.257*(x[ind]-8.) + 0.420*((x[ind]-8.)**2) + 0.374*(x[ind]-8.)**3

    # Case of -values x out of range [0.3,10.0]
    ind = np.where ((x > 10.0) | (x < 0.3))
    a[ind] = 0.0
    b[ind] = 0.0

    #Return Extinction vector
    #Eq 1
    if (Alambda == True):
        return ( 2.5*1./np.log(10.)*( a + b/Rv ) * Av)
    else:
        return ( ( a + b/Rv ) * Av)
    
phot0609 = np.array([19.48,18.90,18.87,19.51])
phot0621 = np.array([19.65,18.97,18.93,19.44])
phot0725 = np.array([20.5,19.7,19.5,19.7])#58689
phot0701 = np.array([19.94,19.18,19.07,19.59])#58665
phot0822 = np.array([20.70,19.98,19.83,20.31])#58717

def analysis(x,y,phot,mask1,mask2,line,custom,xmin,xmax,z = 0.044,rv = 3.1,av = 0.19):
    ebv = av/rv
    new = cardelli(x,av,rv)
    x = x/(1+z)
    y = np.power(10,(0.4*new))*y
    if phot != []:
        x,y,T = mangle(x,y,phot,'no')
    y = np.log10(y) + 25
    plt.figure('Cardelli Mangled')
    plt.plot(x,y, label = phot)
    plt.show()
    return()
    
#def plotter(x,y,add,phot,crop,crop_no,label,z = 0.044,rv = 3.1,av = 0.19):
    
    
    
def plotter(x,y,add,phot,crop,crop_no,label,color,z,ref,rv = 3.1,av = 0.19):

    xlin = np.linspace(3400,7000,1000)
    f = interpolate.interp1d(x, y)
    y = f(xlin) 
    x = xlin
    ebv = av/rv
    new = cardelli(x,av,rv)
    print(len(x))
    x = x/(1+z)
    y = np.power(10,(0.4*new))*y
    y   = y*((x**2)/3e18)
    if phot != []:
        x,y,T = mangle(x,y,phot,'no')
    if phot == []:
        T = None
    if crop == 'pre':
        y = y[crop_no:]
        x = x[crop_no:]
    if crop == 'post':
        y = y[:crop_no]
        x = x[:crop_no]
    logged_y = 2.5*np.log10(y)

    aver = moving_average(200,logged_y)
    if color == 'black':
        logged_y = moving_average(2,logged_y)
        print(len(x))
        plt.plot(x,logged_y + add, label = label, color = 'black', linewidth = 1)
        #plt.plot(x,aver+add)
    if color != 'black':
        y = logged_y  - aver + ref
        plt.plot(x,logged_y + add, color = color, linewidth = 1.5,label=label)
        #plt.plot(x,y+add+4.5,color=color)
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('2.5*Log(flux) + constant')
    plt.xlim(2500,9500)
    plt.yticks([])
    #plt.xlim(3000,7000)
    #plt.ylim(-66.2,-62.7)
    #plt.legend()
    #plt.show()

    return(aver)



    
    
plt.figure()
x,y = spectra_plot('NTT_20190609',0,'none','no')
av = plotter(x,y,-36.2,phot0609,'none',0,'2019hcc','black',0.044,0)
a = np.loadtxt('a.txt') 
b = np.loadtxt('b.txt')*1e-37
plotter(a,b,-40.7,[],'none',0,'tardis','red',0,av)
plt.legend()

plt.axvline(x=4223,linewidth=0.75,linestyle='dashed')
plt.axvline(x=4327,linewidth=0.75,linestyle='dashed')
plt.axvline(x=4431,linewidth=0.75,linestyle='dashed')
plt.axvline(x=4584,linewidth=0.75,linestyle='dashed')
    
plt.show()
    

        