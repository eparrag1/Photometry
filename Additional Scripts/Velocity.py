
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import optimize
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.patches as mpatches
from scipy.stats import chisquare
import spectra_correction as sc

plt.close('all')
Halpha = 6563.3
si = 6355 #from Gutierrez paper
Hbeta = 4861

def func(x, amp, wid,c, cen):
    return c + ((amp) * np.exp(-(x-cen)**2 / (2*wid**2)))

def fit(x,y,cen,width,w):
    a = np.where((x>(cen-width)) & (x<(cen+width)))
    y = y[a]
    x = x[a]
    guess = [-0.75,w,np.mean(y),cen]
    popt,pcov = curve_fit(func,x,y,p0 = guess)
    p_sigma = np.sqrt(np.diag(pcov))
    c = p_sigma[3]
    plt.plot(x,func(x,*popt),label = int(popt[3]))
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Scaled Flux')
    
    centre = popt[3]
    FWHM = abs(2*np.sqrt(2*np.log(2))*popt[1])
    return(centre,c)
    
phot0609 = np.array([19.48,18.90,18.87,19.51])
phot0621 = np.array([19.65,18.97,18.93,19.44])
phot0725 = np.array([20.5,19.7,19.5,19.7])#58689
phot0701 = np.array([19.94,19.18,19.07,19.59])#58665
phot0822 = np.array([20.70,19.98,19.83,20.31])#58717


def velocity(delta,peak,c):        
    v = 3e8*(delta/peak)/1000
    error = 3e8*(c/peak)/1000
    return((v,error))

print('yes')
if __name__ == '__main__':
    w = 15
    plt.figure('H-alpha blueward peak?')
    
    
    x,y = sc.spectra_plot('NTT_20190609',0,'none','no')
    y = sc.moving_average(3,y)
    x,y = sc.plotter(x,y,2,phot0609,'none',0,'06/09/2019','black',0.044)
    
    import pandas as pd
    df = pd.DataFrame(columns=['x', 'y'])
    df['x'] = x
    df['y'] = y
    
    #df.to_csv('/Users/eleonoraparrag/Documents/Spectroscopy/0609_spec.csv',index = False)
    
    
    a,c = fit(x,y,5075,70,w)
    delta = np.abs(a-5169)
    
    print('Fe II')
    print(velocity(delta,5169,c))
    
    
    
    
    data = np.genfromtxt('/Users/eleonoraparrag/Documents/Spectroscopy/2019hcc_20190621_SOAR.txt', dtype=None)
    x =  data[:,0]
    y =  data[:,1]
    y = sc.moving_average(5,y)
    x,y = sc.plotter(x,y,1.7,phot0621,'pre',150,'2019/06/21','black',0.044)
    
    df = pd.DataFrame(columns=['x', 'y'])
    df['x'] = x
    df['y'] = y
    
    #df.to_csv('/Users/eleonoraparrag/Documents/Spectroscopy/0621_spec.csv',index = False)
    
    
    
    a,c = fit(x,y,5062,50,w)
    print('Fe II')
    delta = np.abs(a-5169)
    print(velocity(delta,5169,c))
    a,c = fit(x,y,6216,100,w)
    print('Si II')
    delta = np.abs(a-si)
    print(velocity(delta,si,c))
    print('HV')
    delta = np.abs(a-Halpha)
    print(velocity(delta,Halpha,c))
    print('Halpha')
    a,c = fit(x,y,6395,30,w)
    delta = np.abs(a-Halpha)
    print(velocity(delta,Halpha,c))
    print('Hbeta HV')
    a,c = fit(x,y,4680,30,w)
    delta = np.abs(a-Hbeta)
    print(velocity(delta,Hbeta,c))
    print('Hbeta')
    a,c = fit(x,y,4740,30,w)
    delta = np.abs(a-Hbeta)
    print(velocity(delta,Hbeta,c))
    
    x,y = sc.spectra_plot('NTT_20190701',0,'none','no')
    y = sc.moving_average(1,y)
    x,y = sc.plotter(x,y,1,phot0701,'',0,'2019/07/01','black',0.044)
    
    df = pd.DataFrame(columns=['x', 'y'])
    df['x'] = x
    df['y'] = y
    
    #df.to_csv('/Users/eleonoraparrag/Documents/Spectroscopy/0701_spec.csv',index = False)
    
    a,c = fit(x,y,5062,50,w)
    print('Fe II')
    delta = np.abs(a-5169)
    print(velocity(delta,5169,c))
    a,c = fit(x,y,6262,100,w)
    print('Si II')
    delta = np.abs(a-si)
    print(velocity(delta,si,c))
    print('HV')
    delta = np.abs(a-Halpha)
    print(velocity(delta,Halpha,c))
    print('Halpha')
    a,c = fit(x,y,6395,30,w)
    delta = np.abs(a-Halpha)
    print(velocity(delta,Halpha,c))
    print('Hbeta HV')
    a,c = fit(x,y,4690,30,w)
    delta = np.abs(a-Hbeta)
    print(velocity(delta,Hbeta,c))
    print('Hbeta')
    a,c = fit(x,y,4750,30,w)
    delta = np.abs(a-Hbeta)
    print(velocity(delta,Hbeta,c))
    
    
    
    x,y = sc.spectra_plot('NTT_20190725_6',0,'none','no')
    y = sc.moving_average(1,y)
    x,y = sc.plotter(x,y,1,phot0725,'',0,'2019/07/25','black',0.044)
    
    df = pd.DataFrame(columns=['x', 'y'])
    df['x'] = x
    df['y'] = y
    
    #df.to_csv('/Users/eleonoraparrag/Documents/Spectroscopy/0725_spec.csv',index = False)
    
    
    a,c = fit(x,y,5062,50,w)
    print('Fe II')
    delta = np.abs(a-5169)
    print(velocity(delta,5169,c))
    a,c = fit(x,y,6305,30,w)
    print(c)
    print('Si II')
    delta = np.abs(a-si)
    print(velocity(delta,si,c))
    print('HV')
    delta = np.abs(a-Halpha)
    print(velocity(delta,Halpha,c))
    print('Halpha')
    a,c = fit(x,y,6395,30,w)
    delta = np.abs(a-Halpha)
    print(velocity(delta,Halpha,c))
    print('Hbeta HV')
    a,c = fit(x,y,4681,30,w)
    print(c)
    delta = np.abs(a-Hbeta)
    print(velocity(delta,Hbeta,c))
    print('Hbeta')
    a,c = fit(x,y,4740,35,w)
    print(c)
    delta = np.abs(a-Hbeta)
    print(velocity(delta,Hbeta,c))
    
    
    
    plt.show()
    
    
   
#SN2019hcc

20_hcc = {
        "FeII" : 5870,
        "Halpha" : 7810,
        "Hbeta" : 7640,
        "HVHalpha" : 15750,
        "HVSi" : 6430,
        "HVbeta": 10830
        }
plt.bar(20_hcc.keys(),20_hcc.values())
plt.show()