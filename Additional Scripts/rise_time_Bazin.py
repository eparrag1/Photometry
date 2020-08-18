import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Matern,WhiteKernel,RBF,Product,ConstantKernel,RationalQuadratic,ExpSineSquared
import george
from george.kernels import ExpSquaredKernel, Matern32Kernel
import scipy.optimize as opt
from astropy.table import Table

clipped = pd.read_excel('/Users/eleonoraparrag/Documents/Python/ATLAS/Clipped_data.xlsx', 'Clipped', skiprows = 0, comment='#') 
mean = pd.read_excel('/Users/eleonoraparrag/Documents/Python/ATLAS/Clipped_data.xlsx', 'Mean', skiprows = 0, comment='#') 
full = pd.read_excel('/Users/eleonoraparrag/Documents/Python/ATLAS/Clipped_data.xlsx', 'Full', skiprows = 0, comment='#') 

plt.close('all')
print(clipped.columns)
#ATLAS_curvefit gives more info on the Bazin fit
#Bazin_Mag gives more info on the power law fit
#This is only a summary script to produce the plots

def ATLAS(data):
    o = data[data.Filter.eq('o')]
    b = data[data.Filter.eq('c')]
    
    ATLASo = Table()
    ATLASo['x'] = np.array(o['MJD'])
    ATLASo['y'] = np.array(o['Flux'])
    ATLASo['err'] = np.array(o['Error'])
    
    ATLASb = Table()
    ATLASb['x'] = np.array(b['MJD'])
    ATLASb['y'] = np.array(b['Flux'])
    ATLASb['err'] = np.array(b['Error'])
    
    def Transform(ATLAS,col):
        err = np.abs(2.5*(ATLAS['err']/ATLAS['y'])*(1/np.log(10)))
        mask1 = err > 0.36
        mask2 = err < 0.36
        
        mask = ATLAS['err'] < 30
        flux = ATLAS[mask]
        fluxx,fluxy,fluxerr = flux['x'],flux['y'],flux['err']

        
        lim = ATLAS[mask1]
        full = ATLAS[mask2]
        
        lim['y'] = (-2.5*np.log10(lim['err'] * 10**-6) + 8.9)
        full['err'] =  np.abs(2.5*(full['err']/full['y'])*(1/np.log(10)))
        full['y'] = -2.5*np.log10(full['y'] * 10**-6) + 8.9
        
        less = full['y'] > 19.5
        av = np.mean(full['err'][less])
        limerr = np.ones(len(lim['y']))*av    
        

        x = np.array(full['x'])
        y = np.array(full['y'])
        limx = np.array(lim['x'])
        limy = np.array(lim['y'])
        err = np.array(full['err'])
        return(x,y,err,limx,limy,limerr,fluxx,fluxy,fluxerr)

    x1,y1,err1,limx,limy,limerr,fluxx1,fluxy1,fluxerr1 = Transform(ATLASo,'orange')
    x2,y2,err2,limx2,limy2,limerr2,fluxx2,fluxy2,fluxerr2 = Transform(ATLASb,'blue')
  

    return(x1,y1,err1,limx,limy,limerr,x2,y2,err2,limx2,limy2,limerr2,fluxx1,fluxy1,fluxerr1,fluxx2,fluxy2,fluxerr2)
    

"""----------------------Power Law-------------------------"""
x1,y1,err1,limx,limy,limerr,x2,y2,err2,limx2,limy2,limerr2,fluxx1,fluxy1,fluxerr1,fluxx2,fluxy2,fluxerr2 = ATLAS(full)

fluxx = np.concatenate((fluxx1,fluxx2))
fluxy = np.concatenate((fluxy1,fluxy2))
fluxerr = np.concatenate((fluxerr1,fluxerr2))
x = np.concatenate((x1,limx))
y = np.concatenate((y1,limy))
err = np.concatenate((err1,limerr))
    
    
x,y,err = zip(*sorted(zip(x,y,err)))

def pre_max(t,a,n,texp,c):#c is a constant to correct for not zero
    res = np.zeros(len(t))
    for i in range(len(t)):
        if t[i] < texp:
            res[i] == 0
        if t[i] > texp:
            res[i] = (a*((t[i]-texp)**n))
    return(res + c)
    
from scipy.optimize import curve_fit
x = np.array(x)
y = np.array(y)
err = np.array(err)

pre = np.where(x < 58640)
prex = x[pre]
prey = y[pre]
prey = 25-prey
prerr = err[pre]
popt2,pcov2 = curve_fit(pre_max,prex,prey,p0 = [5e-2,1.5,58617,5],sigma=prerr)
texp = popt2[2]
error_texp = (np.sqrt(np.diag(pcov2)))[2]
xlin = np.linspace(58500,58700)


     
print('THE EXPLOSION EPOCH IS FOUND TO BE ', texp, '+/-',error_texp)
"""----------------------------------------------------------"""

"""-------------------------BAZIN----------------------------"""

def lc(t,A,B,t0,tfall,trise):
    term1 = np.e**(-(t-t0)/tfall)
    term2 = 1 + np.e**((t-t0)/trise)
    return(A*(term1/term2) + B)



from scipy.optimize import curve_fit
x = np.array(fluxx)
y = np.array(fluxy)
err = np.array(fluxerr)


popt,pcov = curve_fit(lc,x,y,p0 = [200,0,58640,-7,6],sigma=err)
xlin2 = np.linspace(58550,58900)
lcy = lc(xlin2,*popt)


A,B,t0,tfall,trise = popt
errA,errB,error_t0,error_tfall,error_trise = np.sqrt(np.diag(pcov))
tmax = t0 + trise*np.log((-trise)/(trise+tfall))
res = trise+tfall
res_positive = np.abs(trise)+np.abs(tfall) 
term4 = np.sqrt(error_trise**2+error_tfall**2) 
res1 = (-trise)/(trise+tfall)
term3 = res1*np.sqrt((error_trise/trise)**2+(term4/res_positive)**2)
res2 = np.log((-trise)/(trise+tfall))
term2 = term3/res1#error in log term
res3 = trise*np.log((-trise)/(trise+tfall))
term1 = res3*np.sqrt((error_trise/trise)**2+(term2/res2)**2)#error in log times trise
error_add = np.sqrt(error_t0**2+term1**2)#error in addition


print('THE MAXIMUM EPOCH IS FOUND TO BE ', tmax, '+/-',error_add)

"""----------------------------------------------------------"""
"""------------------------RISE TIME-------------------------"""


rise = tmax - texp
rise_err = np.sqrt(error_add**2+error_texp**2)

print('RISE TIME IS ', rise,'+/-',rise_err )

"""----------------------------------------------------------"""
"""---------------------FIGURE PLOTTING----------------------"""

from mpl_toolkits.axes_grid.axislines import * 
import matplotlib.gridspec as gridspec

#fig = plt.figure(figsize=(6,4))

#gs = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1])

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (20,10))
    

ax1.plot(xlin,25-pre_max(xlin,*popt2),'r--',color = 'red')
print('values')
print(popt2)
scatter = ax1.scatter(x1, y1, color = 'orange', marker = 'o',alpha=0.25)
ax1.errorbar(x1,y1, yerr = err1, label = 'orange', color = 'orange', fmt = 'o',alpha=0.25)
ax1.errorbar(limx,limy, yerr = limerr,color = 'orange', fmt = 'o',alpha=0.25,marker='v')
ax1.scatter(x2, y2, color = 'blue', marker = 'o',alpha=0.25)
ax1.errorbar(x2,y2, yerr = err2, label = 'cyan', color = 'blue', fmt = 'o',alpha=0.25)
ax1.errorbar(limx2,limy2, yerr = limerr2,color = 'blue', fmt = 'o',alpha=0.25,marker='v')
ax1.legend()
ax1.set_ylabel('Magnitude')
ax1.set_xlabel('MJD')
ax1.invert_yaxis()
ax1.minorticks_on()



ax2.plot(xlin2,lcy)
ax2.errorbar(fluxx1,fluxy1,yerr = fluxerr1, color = 'orange', fmt = 'o',alpha=0.5,label='orange')
ax2.errorbar(fluxx2,fluxy2,yerr = fluxerr2, color = 'blue', fmt = 'o',alpha=0.5,label = 'cyan')
ax2.legend()
ax2.set_ylabel('Flux')
ax2.set_xlabel('MJD')
ax2.minorticks_on()
#rc('text', usetex=True)
#rc('font',family='Times New Roman')
#rc('xtick', labelsize=13)
#rc('ytick', labelsize=13)

