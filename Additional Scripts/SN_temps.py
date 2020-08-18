import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

directory = '/Users/eleonoraparrag/Documents/Spectroscopy/'
#plt.close('all')
def flux(mags,error,bands):
    
    def bbody(lam,T,A):
        Blam = A*(2*6.626e-27*(3e10)**2/(lam*1e-8)**5)/(np.exp(6.627e-27*3e10/(lam*1e-8*1.38e-16*T))-1)
        return(Blam)


    # effective wavelengths of filters:
    wle = {'S':2030,'D':2231,'P':2634,'u': 3560 , 'g': 4830 , 'r':6260 , 'i': 7670 , 'z': 9100 
           , 'U': 3600 , 'B': 4380 , 'V': 5450 , 'X': 6410 , 'Y': 7980, 'J':12200, 'H':16300,
           'K': 21900, 'G': 4718.9, 'R': 6185.2, 'I': 7499.8, 'Z': 8961.5}
    #SLOAN
    #'G': 4718.9, 'R': 6185.2, 'I': 7499.8, 'Z': 8961.5
    #'G': 541.4, 'R': 247.2, 'I': 138.3, 'Z': 81.5
    #I've renamed R and I to X and Y, to replace
     
     
    # Reference fluxes for converting magnitudes to flux
    zp = {'S': 536.2, 'D': 463.7, 'P': 412.3 ,'u': 859.5 , 'g': 466.9 , 'r': 278.0 , 'i': 185.2 
          , 'z': 131.5 , 'U': 417.5 , 'B': 632 , 'V': 363.1 , 'X': 217.7 , 'Y': 112.6, 'J':31.47
          , 'H':11.38, 'K':3.961, 'G': 541.4, 'R': 247.2, 'I': 138.3, 'Z': 81.5}
    
    wl = []
    fref = []
    
    #mags = np.array([20.26,18.64,18.79,18.41]) 
    
    
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
    error = np.abs(flux)*np.abs(0.4*np.log(10)*error)
    wl = np.array(wl)
    flux = np.array(flux/1e43)
    error = np.array(error/1e43)

    
    BBparams, covar = curve_fit(bbody,wl,flux)
    
    T = BBparams[0]

    return(wl,flux,error)

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

def blackbody_curve_fit(x,y,err):
    popt,cov  = curve_fit(blackbody,x,y,sigma = err,p0=(7000,7e8),absolute_sigma = True)
    Terr = np.sqrt(np.diag(cov))[0]
    bb = blackbody(x,*popt)
    T,b = popt
    return(T,b,Terr,bb)
    
    
def mangle(mags,mags_error,bands):
    wl,fref,error = flux(mags,mags_error,bands)
    T,b,Terr,bb=  blackbody_curve_fit(wl,fref,error)
    wl,bb,fref = zip(*sorted(zip(wl,bb,fref)))
    plt.figure()
    plt.plot(wl,bb, label = T)
    plt.errorbar(wl,fref,yerr=error,fmt='o',label=Terr)
    plt.legend()
    plt.show()
    return(T,Terr)
    
print('SN2019hcc, this is my first temp measure but a bit later - 8242+/-1340')   
#LSQ14mo - I need more bands I think
excel = '/Users/eleonoraparrag/Documents/Spectroscopy/Spreadsheets/SN_temperatures.xlsx'
xl = pd.ExcelFile('/Users/eleonoraparrag/Documents/Spectroscopy/Spreadsheets/SN_temperatures.xlsx')
sheets = xl.sheet_names
for sheet in ('SN2015bn_1','SN2015bn_2','iPTF16bad','PTF12dam','LSQ14mo','SN2010aj','SN2013ej','SN2008fq','SN2014G','SN2010kd_1','SN2010kd_2'):
    print(sheet)
    data = pd.read_excel(excel, sheet, skiprows = 0, comment='#') 
    data.columns = ['bands','mag','mag_error']
    if sheet == 'iPTF16bad':
        g = data[data.bands.eq('g')]
        r = data[data.bands.eq('r')]
        i = data[data.bands.eq('i')]
        mags = np.array([float(g['mag']),float(r['mag']),float(i['mag'])])
        mags_err = np.array([float(g['mag_error']),float(r['mag_error']),float(i['mag_error'])])
        T,Terr = mangle(mags,mags_err,'gri')
    if sheet in ('LSQ14mo'):
        g = data[data.bands.eq('g')]
        r = data[data.bands.eq('r')]
        i = data[data.bands.eq('i')]
        u = data[data.bands.eq('u')]
        mags = np.array([float(g['mag']),float(r['mag']),float(i['mag']),float(u['mag'])])
        mags_err = np.array([float(g['mag_error']),float(r['mag_error']),float(i['mag_error']),float(u['mag_error'])])
        T,Terr = mangle(mags,mags_err,'griu')
    if sheet in ('SN2010aj','SN2014G','SN2010kd_1','SN2010kd_2'):
        B = data[data.bands.eq('B')]
        V = data[data.bands.eq('V')]
        R = data[data.bands.eq('X')]
        I = data[data.bands.eq('Y')]
        mags = np.array([float(B['mag']),float(V['mag']),float(R['mag']),float(I['mag'])])
        mags_err = np.array([float(B['mag_error']),float(V['mag_error']),float(R['mag_error']),float(I['mag_error'])])
        T,Terr = mangle(mags,mags_err,'BVXY')
    if sheet not in ('iPTF16bad','LSQ14mo','SN2010aj','SN2014G','SN2010kd_1','SN2010kd_2'):  
        B = data[data.bands.eq('B')]
        V = data[data.bands.eq('V')]
        g = data[data.bands.eq('g')]
        r = data[data.bands.eq('r')]
        i = data[data.bands.eq('i')]
        mags = np.array([float(B['mag']),float(V['mag']),float(g['mag']),float(r['mag']),float(i['mag'])])
        mags_err = np.array([float(B['mag_error']),float(V['mag_error']),float(g['mag_error']),float(r['mag_error']),float(i['mag_error'])])
        T,Terr = mangle(mags,mags_err,'BVgri')
    print(sheet,T,Terr)
    print('NEXT')
