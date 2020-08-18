
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import optimize
from scipy.optimize import curve_fit
from scipy import interpolate
import pandas as pd

#plt.close('all')
def flux(mags,bands,error):
    
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
    
    flux = 10**(-0.4*mags)
    error = 10**(-0.4*mags)*np.abs(0.4*np.log(10)*error)
    cons = 4*np.pi*SN_distance**2*fref
    flux = cons*flux
    error = error*cons

    wl = np.array(wl)
    flux = flux/1e43
    error= error/1e43

    
    BBparams, covar = curve_fit(bbody,wl,flux)
    T = BBparams[0]

    return(wl,flux,error)
    
    
def smartint(x,y,xref,yref):
    ir = (xref>=min(x))&(xref<=max(x))#where the reference is within x
    yint = interpolate.interp1d(x[np.argsort(x)],y[np.argsort(x)])(xref[ir])#interpolate x and y to range of xref within
    yord = np.zeros(len(xref),dtype=float)#empty array length xref
    ylow = yint[np.argmin(xref[ir])]-yref[ir][np.argmin(xref[ir])]+\
           yref[xref<min(x)]#lowest interpolated y - lowest yref , 
    yup  = yint[np.argmax(xref[ir])]-yref[ir][np.argmax(xref[ir])]+\
           yref[xref>max(x)]
    yord[ir] = yint #within range of xref, keep interpolated
    yord[xref<min(x)] = ylow #for xref out of range of x, take y as yref here 
    #(the lowest interp y  - lower yref within range) + yref  - simple
    yord[xref>max(x)] = yup #same as before
    return yord #basically extend x and y to cover xref range

def SED(data_xls,bands):
    _bandref = 'B'
    _err = 0.1
    ph,mag,mag_err = {},{},{}
        
    for b in bands:
        #_nol = np.array(sndata['source'][b])>=0 #wtf is source? where it is greater than 0
        mag[b] = np.array(data_xls[data_xls['bands'] == b]['mag'])
        _mag_err = np.array(data_xls[data_xls['bands'] == b]['mag_error'])
        #source = np.array(sndata['source'][b])[_nol] #only source greater than 0
        #ph[b] = np.array(sndata['jd'][b])[_nol]-sndata['jdmax'][_bandref]
        ph[b] = np.array(data_xls[data_xls['bands'] == b]['MJD'])
        #julian date where source is greater than zero 
        #max jd in the reference band?
        #align all to reference band - ignore for now
    
        mag_err[b] = np.where(_mag_err>0,_mag_err,_err) #if mag_err is less than zero replace with 0.1 
    
    magint,magint_err = {},{}
    
    for b in bands:
        xadd,yadd = [],[]
        if b != _bandref: #if it is not the reference band
            y1,y2 = min(mag[_bandref]),max(mag[_bandref])   
            if y1>min(mag[b]): y1=min(mag[b])
            if y2<max(mag[b]): y2=max(mag[b])#if mags stray out of reference band range, replace 
            magint[b] = smartint(ph[b],mag[b],ph[_bandref],mag[_bandref])#extend band to bandref range
            magint_err[b] = smartint(ph[b],mag_err[b],ph[_bandref],np.zeros(len(mag[_bandref]),dtype=float))        
        else:
            phint = ph[b]
            magint[b] = mag[b]
            magint_err[b] = mag_err[b]
    return(magint,magint_err,ph)
    
    
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
    T,b = popt
    return(T,b,Terr)
        
def blackbody_fit(x,y):
    guess = 5000
    b = 7e8
    def minfunc(A):
        T,b = A
        return sum((y-blackbody(x,T,b))**2)
    
    T,b = optimize.fmin(minfunc,(guess,b))
    bb = blackbody(x,T, b)
    return(T,b)
    
def go(magint,magint_err,ph,bands,texp):
    Temps = np.array([])
    Temps_err = np.array([])
    for i in range(len(ph['B'])):               
        mags = np.array([])
        mag_errs = np.array([])
        for b in bands:

            mags = np.append(mags,magint[b][i])
            mag_errs = np.append(mag_errs,magint_err[b][i])

        wl,fref,err = flux(mags,bands,mag_errs)

        wl,fref,err = zip(*sorted(zip(wl,fref,err)))
        #Extrapolate out to 3000-10000 range?
        #wl,fref = inter(wl,fref,np.linspace(3000,10000,100))     
        #BLACKBODY
 
        T,b,Terr = blackbody_curve_fit(np.array(wl),np.array(fref),np.array(err))
        Temps = np.append(Temps,T)
        Temps_err = np.append(Temps_err,Terr)
        #bb_x = np.linspace(4500,7500,100)
        #bb = blackbody(bb_x,T, b)
        #plt.figure()
        #plt.plot(bb_x,bb)
        #plt.errorbar(wl,fref,yerr = err, fmt='o')
        #plt.show()
        
    ph['B'] = ph['B'] - texp 

    return(ph['B'],Temps,Temps_err)


if __name__ == '__main__':
 

    f, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize = (20,10))
    
    excel = ('/Users/eleonoraparrag/Documents/Python/Photometry/SNdata.xlsx')

    data_xls = pd.read_excel(excel, skiprows = 0, comment='#',sheetname='SN2013ej')
    magint,magint_err,ph = SED(data_xls,'BVJKHugriz')
    A,B,C = go(magint,magint_err,ph,'BVJKHugriz',56498)
    ax1.errorbar(A,B,yerr=C,fmt='o',label = 'SN2013ej',color='tab:green')
    
    excel = ('/Users/eleonoraparrag/Documents/Python/Photometry/SNdata.xlsx')
    data_xls = pd.read_excel(excel, skiprows = 0, comment='#',sheetname='SN2019hcc_rp')
    magint,magint_err,ph = SED(data_xls,'BVgirGRIZ')
    A,B,C = go(magint,magint_err,ph,'BVgirGRIZ',58619)
    ax1.errorbar(A,B,yerr=C,fmt='o',label = 'SN2019hcc',color='black')
    
    #R and I replaced with X and Y, to allow SDSS filters
    data_xls = pd.read_excel(excel, skiprows = 0, comment='#',sheetname='SN2014G')
    magint,magint_err,ph = SED(data_xls,'BVXYU')
    A,B,C = go(magint,magint_err,ph,'BVXYU',56669.6)
    ax1.errorbar(A,B,yerr=C,fmt='o',label = 'SN2014G',color = 'tab:red')
    
    
    data_xls = pd.read_excel(excel, skiprows = 0, comment='#',sheetname='SN2010aj')
    magint,magint_err,ph = SED(data_xls,'BVUXYir')
    A,B,C = go(magint,magint_err,ph,'BVUXYir',55267)
    ax1.errorbar(A,B,yerr=C,fmt='o',label = 'SN2010aj',color='tab:blue')
    
    data_xls = pd.read_excel(excel, skiprows = 0, comment='#',sheetname='SN2009dd')
    magint,magint_err,ph = SED(data_xls,'BVUXYir')
    A,B,C = go(magint,magint_err,ph,'BVUXYir',54934)
    ax1.errorbar(A,B,yerr=C,fmt='o',label = 'SN2009dd',color='tab:orange')
    
    
    data_xls = pd.read_excel(excel, skiprows = 0, comment='#',sheetname='SN2008fq')
    magint,magint_err,ph = SED(data_xls,'BVJHgriu')
    A,B,C = go(magint,magint_err,ph,'BVJHgriu',54724)
    ax1.errorbar(A,B,yerr=C,fmt='o',label = 'SN2008fq',color='tab:purple')
    
    ax1.set_ylabel('Temperature (K)')
    ax1.minorticks_on()
    ax1.set_xlim(35,107)
    ax1.legend()
    
    
    
    excel = ('/Users/eleonoraparrag/Documents/Python/Photometry/SNdata.xlsx')
    data = pd.read_excel(excel, 'SN2013ej', skiprows = 0, comment='#') 
    data['MJD'] = np.floor(data['MJD'])                
    B = data[data.bands.eq('B')]
    V = data[data.bands.eq('V')]
    g = data[data.bands.eq('g')]
    r = data[data.bands.eq('r')]
    BV = B.merge(V, how = 'inner', on = ['MJD'])
    gr = g.merge(r, how = 'inner', on = ['MJD'])
    BVerr = np.sqrt(np.array(BV['mag_error_x']**2)+np.array(BV['mag_error_y']**2))
    grerr = np.sqrt(np.array(gr['mag_error_x']**2)+np.array(gr['mag_error_y']**2))
    ax2.errorbar(np.array(BV['MJD'])-56498,np.array(BV['mag_x'])-np.array(BV['mag_y']),yerr = BVerr,fmt='o',label = 'SN2013ej',color='tab:green')
    ax3.errorbar(np.array(gr['MJD'])-56498,np.array(gr['mag_x'])-np.array(gr['mag_y']),yerr = grerr,fmt='o',label = 'SN2013ej',color='tab:green')
    
    
    
    
    
    excel = ('/Users/eleonoraparrag/Documents/Python/Photometry/SNdata.xlsx')
    data = pd.read_excel(excel, 'SN2014G', skiprows = 0, comment='#') 
    B = data[data.bands.eq('B')]
    V = data[data.bands.eq('V')]
    BV = B.merge(V, how = 'inner', on = ['MJD'])
    BVerr = np.sqrt(np.array(BV['mag_error_x']**2)+np.array(BV['mag_error_y']**2))
    ax2.errorbar(np.array(BV['MJD'])-56669.6,np.array(BV['mag_x'])-np.array(BV['mag_y']),yerr = BVerr,fmt='o',label = 'SN2014G',color = 'tab:red')
    
    
    excel = ('/Users/eleonoraparrag/Documents/Python/Photometry/SNdata.xlsx')
    data = pd.read_excel(excel, 'SN2010aj', skiprows = 0, comment='#') 
    B = data[data.bands.eq('B')]
    V = data[data.bands.eq('V')]
    BV = B.merge(V, how = 'inner', on = ['MJD'])
    BVerr = np.sqrt(np.array(BV['mag_error_x']**2)+np.array(BV['mag_error_y']**2))
    ax2.errorbar(np.array(BV['MJD'])-55267,np.array(BV['mag_x'])-np.array(BV['mag_y']),yerr = BVerr,fmt='o',label = 'SN2010aj',color='tab:blue')
    
    
    excel = ('/Users/eleonoraparrag/Documents/Python/Photometry/SNdata.xlsx')
    data = pd.read_excel(excel, 'SN2009dd', skiprows = 0, comment='#') 
    B = data[data.bands.eq('B')]
    V = data[data.bands.eq('V')]
    BV = B.merge(V, how = 'inner', on = ['MJD'])
    BVerr = np.sqrt(np.array(BV['mag_error_x']**2)+np.array(BV['mag_error_y']**2))
    ax2.errorbar(np.array(BV['MJD'])-54934,np.array(BV['mag_x'])-np.array(BV['mag_y']),yerr = BVerr,fmt='o',label = 'SN2009dd',color='tab:orange')
    
    
        
    excel = ('/Users/eleonoraparrag/Documents/Python/Photometry/SNdata.xlsx')
    data = pd.read_excel(excel, 'SN2008fq', skiprows = 0, comment='#') 
    B = data[data.bands.eq('B')]
    V = data[data.bands.eq('V')]
    g = data[data.bands.eq('g')]
    r = data[data.bands.eq('r')]
    
    BV = B.merge(V, how = 'inner', on = ['MJD'])
    gr = g.merge(r, how = 'inner', on = ['MJD'])
    
    BVerr = np.sqrt(np.array(BV['mag_error_x']**2)+np.array(BV['mag_error_y']**2))
    grerr = np.sqrt(np.array(gr['mag_error_x']**2)+np.array(gr['mag_error_y']**2))
    
    
    ax2.errorbar(np.array(BV['MJD'])-54724,np.array(BV['mag_x'])-np.array(BV['mag_y']),yerr = BVerr,fmt='o',label = 'SN2008fq',color='tab:purple')
    ax2.set_ylabel('B - V')
    ax2.minorticks_on()
    ax2.set_xlim(35,107)
    
    ax3.errorbar(np.array(gr['MJD'])-54724,np.array(gr['mag_x'])-np.array(gr['mag_y']),yerr = grerr,fmt='o',label = 'SN2008fq',color='tab:purple')
    ax3.set_ylabel('g-r')
    ax3.set_xlabel('Days past explosion')
    ax3.minorticks_on()
    ax3.set_xlim(35,107)
    
    
        
    excel = ('/Users/eleonoraparrag/Documents/Python/Photometry/SNdata.xlsx')
    data = pd.read_excel(excel, 'SN2019hcc_rp', skiprows = 0, comment='#') 
    B = data[data.bands.eq('B')]
    V = data[data.bands.eq('V')]
    g = data[data.bands.eq('g')]
    r = data[data.bands.eq('r')]
    
    BV = B.merge(V, how = 'inner', on = ['MJD'])
    gr = g.merge(r, how = 'inner', on = ['MJD'])
    
    BVerr = np.sqrt(np.array(BV['mag_error_x']**2)+np.array(BV['mag_error_y']**2))
    grerr = np.sqrt(np.array(gr['mag_error_x']**2)+np.array(gr['mag_error_y']**2))
    
    
    ax2.errorbar(np.array(BV['MJD'])-58619,np.array(BV['mag_x'])-np.array(BV['mag_y']),yerr = BVerr,fmt='o',label = 'SN2019hcc',color='black')
    ax2.set_ylabel('B - V')
    ax2.minorticks_on()
    ax2.set_xlim(35,107)
    
    ax3.errorbar(np.array(gr['MJD'])-58619,np.array(gr['mag_x'])-np.array(gr['mag_y']),yerr = grerr,fmt='o',label = 'SN2019hcc',color='black')
    ax3.set_ylabel('g-r')
    ax3.set_xlabel('Days past explosion')
    ax3.minorticks_on()
    ax3.set_xlim(35,107)
    
    
    
    
    
    ax2.legend()
    ax3.legend()
    plt.show()