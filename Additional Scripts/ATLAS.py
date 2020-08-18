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

#plt.close('all')
print(clipped.columns)
            
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
        #mask3 = np.floor(ATLAS['x']) == 58785
        #ATLAS = ATLAS[mask3]
        err = np.abs(2.5*(ATLAS['err']/ATLAS['y'])*(1/np.log(10)))
        mask1 = err > 0.36
        mask2 = err < 0.36
        lim = ATLAS[mask1]
        full = ATLAS[mask2]
        
        #lim = ATLAS
        #full = ATLAS
        n = 2
        lim['y'] = (-2.5*np.log10(n * lim['err'] * 10**-6) + 8.9)
        full['err'] =  np.abs(2.5*(full['err']/full['y'])*(1/np.log(10)))
        full['y'] = -2.5*np.log10(full['y'] * 10**-6) + 8.9
        
        scatter = plt.scatter(full['x'],full['y'], color = col, marker = 'o')
        plt.errorbar(full['x'],full['y'], yerr = full['err'], label = col, color = col, fmt = 'o')
        scatter = plt.scatter(lim['x'],lim['y'], color = col, marker = 'v')
        x = np.array(full['x'])
        y = np.array(full['y'])
        limx = np.array(lim['x'])
        limy = np.array(lim['y'])
        err = np.array(full['err'])
        return(scatter,x,y,err,limx,limy)

    scatter,x1,y1,err1,limx,limy = Transform(ATLASo,'orange')
    scatter,x2,y2,err2,limx,limy = Transform(ATLASb,'blue')
    #x = np.concatenate((x1,x2))
    #y = np.concatenate((y1,y2))
    #err = np.concatenate((err1,err2))

    
    #Spectra Epochs
    """
    plt.axvline(x=58643, linewidth = 0.5, alpha = 0.5)
    plt.axvline(x=58655, linewidth = 0.5, alpha = 0.5)
    plt.axvline(x=58665, linewidth = 0.5, alpha = 0.5)
    plt.axvline(x=58689, linewidth = 0.5, alpha = 0.5)
    plt.axvline(x=58717, linewidth = 0.5, alpha = 0.5)
    plt.axvline(x=58814, linewidth = 0.5, alpha = 0.5)
    """
    

    ax = scatter.axes
    ax.invert_yaxis()
    plt.legend()
    plt.ylabel('Magnitude')
    plt.xlabel('Days from Explosion')
    plt.show()
    return(x1,y1,err1)
ATLAS(full)

