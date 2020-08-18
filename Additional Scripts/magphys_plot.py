#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:27:26 2020

@author: eleonoraparrag
"""
import numpy as np
plt.close('all')

"""
df = pd.read_csv('4732.sed',skiprows=10,delim_whitespace=True)
plt.figure('4732')
plt.plot(df['lg(lambda/A)'],df['Attenuated'])
plt.xlabel('lg(lambda/A)')
plt.ylabel('Units?')
plt.show()
"""

df = pd.read_csv('SN2019hc.s',skiprows=10,delim_whitespace=True)
plt.figure('SN2019hcc')
plt.xlim(700,10000)
#plt.ylim(3.8,5.2)
wav = 10**df['lg(lambda/A)']
plt.plot(wav,(df['Attenuated']),label='Attenuated')
#plt.plot(wav,(df['Unattenuated']),label='Attenuated')
#plt.plot(wav,(10**df['Attenuated'])*wav*3.846e33,label='Attenuated')
#plt.plot(wav,(10**df['Unattenuated'])*wav*3.846e33,label='Unattenuated')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('log(L_lambda/L_sun A-1)')

plt.legend()
plt.show()

def convert(fl,flerr,wav,d):
    dist = 4*np.pi*(d*1e6*3.0857e16)**2
    print(dist)
    Jy = fl*1e-26
    Jyerr = flerr*1e-26
    print(dist*Jy)
    nu_conv = 3e8/wav**2
    L_lambda = dist*Jy*nu_conv*1e-10
    L_lambda_err = dist*Jyerr*nu_conv*1e-10
    print('L_lambda')
    print(L_lambda,L_lambda_err)
    print('L_lambda/Lsun')
    print(L_lambda/3.846e26)
    print(np.log10(L_lambda/3.846e26))
    logerr = L_lambda_err/(L_lambda*np.log(10))
    print(logerr)
    log = np.log10(L_lambda/3.846e26)
    return(log,logerr)
    
convert(19e-6,1.403E-06,0.460e-6,194.77)
wav = np.array([0.460,0.472,0.538,0.619,0.750,0.896])
fl = np.array([19.00E-06,25.77E-06,34.52E-06,19.66E-06,21.60E-06,19.44E-06])
flerr = np.array([1.403E-06,2.941E-06,3.186E-06,2.202E-06,1.584E-06,1.885E-06])
d=194.77
wav_m = wav*1e-6
converted,converr = convert(fl,flerr,wav_m,d)

L_flux = np.log10((1.044*fl*3e14)/wav) #from sef_fit.pro??
converted_2 = (10**converted)*wav*1e4*3.846e33
#plt.scatter(wav*1e4,converted)
plt.errorbar(wav*1e4,converted,yerr=converr,fmt='o',label = 'Photometry')
plt.legend()





#l = ((4*np.pi*((d*3.0857e16)**2)*fl)/(3.846e33*1e10))

#d = d*3.0857e16
#l=fl*1e-23*3.283608731e-33*(d**2)
