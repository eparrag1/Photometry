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
import spectra_correction as sp
import cachito as cac

plt.close('all')

def excel_II(name,z,av,pre,add):
    excel = '/Users/eleonoraparrag/Documents/Spectroscopy/Final_comp/moderate.xlsx'
    data_xls = pd.read_excel(excel, name, skiprows = 0, comment='#') 
    data_xls.columns = ['Wav','Flux']
    x,y = np.array(data_xls['Wav']),np.array(data_xls['Flux'])
    #y = sp.moving_average(av,y) 
    x,y = sp.plotter(x,y,add,[],'pre',pre,name,'black',z)
    return(x,y)

def vels(x,y,guess):
    a,c = cac.fit(x,y,guess[0],guess[1],guess[2])
    print('Fe II')
    print(a,c)
    delta = np.abs(a-5169)
    print(cac.velocity(delta,5169,c))
    a,c = cac.fit(x,y,guess[3],guess[4],guess[5])
    print('Halpha')
    print(a,c)
    a,c = cac.fit(x,y,guess[6],guess[7],guess[8])
    delta = np.abs(a-6563.3)
    print(cac.velocity(delta,6563.3,c))
    print('Hbeta')
    print(a,c)
    a,c = cac.fit(x,y,guess[9],guess[10],guess[11])
    delta = np.abs(a-4861)
    print(cac.velocity(delta,4861,c))


x,y = excel_II('SN2014G_2',0.0039,5,100,8e-15)
guess = [5030,30,20,6070,100,15,6372,30,10,4725,50,15]
vels(x,y,guess)   
x,y = excel_II('SN2014G',0.0039,5,100,4e-15)
guess = [5055,70,30,6100,100,15,6392,50,15,4735,50,20]
vels(x,y,guess)
x,y = excel_II('SN2014G_3',0.0039,5,100,0)
guess = [5090,300,40,6100,100,15,6399,50,35,4755,50,35]
vels(x,y,guess)

"""
plt.figure('Ignore')
x1,y1 = excel_II('SN2014G_2',0.0039,5,100,8e-15)
x2,y2 = excel_II('SN2014G',0.0039,5,100,4e-15)
x3,y3 = excel_II('SN2014G_3',0.0039,5,100,0)
plt.show()

plt.figure()
x1 = 3e8*((x1-6563.3)/6563.3)/1000
plt.plot(x1,y1)
print(cac.fit(x1,y1,-8600,1100,600))

x2 = 3e8*((x2-6563.3)/6563.3)/1000
plt.plot(x2,y2)
print(cac.fit(x2,y2,-8400,1100,600))
x3 = 3e8*((x3-6563.3)/6563.3)/1000
plt.plot(x3,y3)
print(cac.fit(x3,y3,-7000,2000,1000))

plt.show()
"""

