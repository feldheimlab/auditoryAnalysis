#!/usr/bin/env python3

'''
Tools for fitting datat to common distributions.

Author: Brian Mullen
Date: June 2023

'''

import matplotlib.pyplot as plt
import numpy as np

from scipy import optimize
from scipy import stats
from scipy import special

def chiSquaredTest(data, modelfit, params, plot=False, savepath=None):
    
    normdata = np.sum(data)*modelfit/np.sum(modelfit)
    
    chiresults = stats.chisquare(data, normdata)
    df = data.size-len(params)
    print(chiresults)
    print('Chi squared per degrees freedom: ', chiresults.statistic/df)
    if plot:
        try:
            chibins = np.arange(0,df*3,0.1)
        except:
            chibins = np.arange(0,100,0.1)
            
        fig, axs = plt.subplots(1,2)
        axs[0].title.set_text('Chi2 test')
        axs[0].set_ylabel('probability')
        axs[0].set_xlabel('Chi2 statistic')

        axs[0].plot(chibins, stats.chi2.pdf(chibins, df=df))
        axs[0].vlines(chiresults.statistic,0,0.05, 'r')
        axs[0].text(chiresults.statistic,0.05, 'p-value: {}'.format(np.round(chiresults.pvalue,2)))
        axs[1].scatter(data,  modelfit, c='k')
        maxval = np.max(data)
        if any(maxval < modelfit):
            maxval = np.max(modelfit)
        axs[1].set_xlim(0,maxval)
        axs[1].set_ylim(0,maxval)
        axs[1].plot([0, maxval],[0, maxval], linestyle=':', color='k')
        axs[0].title.set_text('Chi2 test')
        axs[1].set_ylabel('fit values')
        axs[1].set_xlabel('actual values')
        plt.tight_layout()
        if savepath is not None:
            print('saving to :', savepath)
            plt.savefig(savepath, dpi=300)
        plt.show()

# uniform distribution
def uniform(height, X):
    return height*np.ones_like(X)

def uniformParams(data, X):
    height = np.mean(data)
    return height

def fitUniform(data, X):
    '''
    Returns (height, xmean, xsd)
    '''
    params =  uniformParams(data, X)
    error_function = lambda p: np.ravel(uniform(*p, X) - data)
    p, success = optimize.leastsq(error_function, params)
    
    return p


'''
# Example code on usage: 

data = np.array([1,0,1,0,1,2,2,4,13,24,14,10,6,3,1,1,3,2,2,1])
bins = np.arange(0,data.shape[0],1)
interp = np.arange(0,data.shape[0],.1)

#data
p = fitUniform(data, bins)
fit = uniform(*p, bins)

#print the fit parameters
param_labels = ['height']
for i, label in enumerate(param_labels):
    print(label, ':', np.round(p[i], 4))

#graph the data
plt.bar(bins, data)
plt.xlabel('time (ms)')
plt.ylabel('sum AP events')

plt.plot(interp, uniform(*p, interp), linestyle=':', c='black')
plt.show()

chiSquaredTest(data, uniform(*p, bins))

'''

# Gaussian with uniform background
def gaussian(height, xmean, xsd, base):
    return lambda x: height*np.exp(-(((xmean-x)/xsd)**2))+base

def gaussianmoments(data):
    '''
    Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 1D distribution
    '''
    base = np.mean(data[:5])
    datam = data
    datam[datam<0]=0
    total = datam.sum()
    X = np.indices(data.shape)

    xmean = (X*datam).sum()/total
    xsd = np.sqrt(np.abs(np.sum((X-xmean)**2*data)/np.sum(data)))
    height = datam.max()
    
    return height, xmean, xsd, base

def fitGaussian(data):
    '''
    Returns (height, xmean, xsd, base)
    '''
    params = gaussianmoments(data)
    error_function = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(error_function, params)
    
    return p

'''
# Example code on usage: 

data = np.array([1,0,1,0,1,2,2,4,13,24,14,10,6,3,1,1,3,2,2,1])

bins = np.arange(0,data.shape[0],1)
interp = np.arange(0,data.shape[0],.1)

#data
p = fitGaussian(data)
fit = gaussian(*p)

#print the fit parameters
param_labels = ['height', 'mean', 'std', 'base']
for i, label in enumerate(param_labels):
    print(label, ':', np.round(p[i], 4))

#graph the data
plt.bar(bins, data)
plt.xlabel('time (ms)')
plt.ylabel('sum AP events')

plt.plot(interp, fit(interp), linestyle=':', c='black')
plt.show()


res = chiSquaredTest(data, fit(bins))

'''


# Vonm Mises 
def vonMises(height, xmean, conc):
    '''
    x in radians
    '''
    return lambda x: height*np.exp(conc*np.cos(x-xmean))/(2*np.pi*special.jv(0, conc))

def vonmisesmoments(data, X):
    '''
    X in radians
    Returns (height, xmean, conc)
    the vonMises parameters of a 1D distribution
    '''
    
    datam = data
    datam[datam<0]=0
    total = datam.sum()

    xmean = (X*datam).sum()/total
    conc = np.sqrt(np.abs(np.sum((X-xmean)**2*data)/np.sum(data)))
    height = datam.max()
    
    return height, xmean, conc

def fitVonMises(data, X):
    '''
    X in radians
    Returns (height, xmean, conc)
    '''
    params = vonmisesmoments(data, X)
    error_function = lambda p: np.ravel(vonMises(*p)(X) - data)
    p, success = optimize.leastsq(error_function, params)
    
    return p


'''
# Example code on usage: 

data = np.array([ 6.,  7.,  7.,  3.,  6.,  9., 10.,  8.,  7., 20., 14., 17., 14., 18.,  9.,  7., 12.])
azim = np.array([-144, -126, -108, -90, -72, -54, -36, -18, 0, 18, 36, 54, 72, 90, 108, 126, 144])
azimrad = azim * (np.pi/180)
interp = np.arange(azimrad[0],azimrad[-1],.1)

#data
p = fitVonMises(data, azimrad)
fit = vonMises(*p)

#print the fit parameters
param_labels = ['height', 'mean', 'std']
for i, label in enumerate(param_labels):
    print(label, ':', np.round(p[i], 4))

#graph the data
plt.bar(azimrad, data, width=(np.abs(np.min(azimrad))+np.abs(np.max(azimrad)))/len(data))
plt.xlabel('Azimuth (radians)')
plt.ylabel('sum AP events')

plt.plot(interp, fit(interp), linestyle=':', c='black')
plt.show()


res = chiSquaredTest(data, fit(azimrad))
'''


# Kent distribution

def rodrot(targetvector, rotationaxis, angle):
    # this function does rotation of a vector in 3d space accordingly to
    # Rodrigues rotation formula.
    
    r1 = targetvector*np.cos(angle)
    r2 = np.cross(rotationaxis, targetvector) * np.sin(angle)
    r3 = rotationaxis * (np.transpose(rotationaxis) * targetvector) * (1 - np.cos(angle))
     
    return np.squeeze(r1 + r2 + r3)


def sphericalUnit(theta, phi):
    # this function gives a unit vectors of spherical coordinates.
    # the notation is based on Arfken
    # theta is polar angle
    # phi is azimuthal angle.

    st = np.sin(theta);
    ct = np.cos(theta);
    sp = np.sin(phi);
    cp = np.cos(phi);

    unitvecs = np.array([[st * cp,  ct * cp, -sp],
                [st * sp,  ct * sp,  cp],
                [ct,      -st,      0]])
    
    return unitvecs

def sph2cart(theta, phi):
    # this returns cartesian coord based on the spherical coordinates.
    # this assumes a unit circle
    return [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]

def kent(xyz, 
         height, 
         beta, kappa, gamma1, gamma2, gamma3):
#, base):
    kent_dist = height * np.exp(-kappa) * np.exp(kappa * np.dot(xyz, gamma1) + 
            beta * kappa * (np.dot(xyz, gamma2)**2 - np.dot(xyz, gamma3)**2)) 
#- base 
    return np.squeeze(kent_dist)

def kentdist(kappa, beta, theta, phi, alpha, height, 
             #base, 
             xyz):

    units = sphericalUnit(theta, phi)
    gamma1 = units[:, 0]

    gamma2 = rodrot(units[:, 1], units[:, 0], alpha)
    gamma3 = rodrot(units[:, 2], units[:, 0], alpha)
    
    return kent(xyz, height, beta, kappa, gamma1, gamma2, gamma3)
    #, base)

def kentRandStart():
    kappa = np.random.uniform(low=0, high=100, size=1)
    beta = np.random.uniform(low=-0.5, high=0.5, size=1)
    theta = np.random.uniform(low=0, high=np.pi/2, size=1)
    phi =  np.random.uniform(low = -144 / 180 * np.pi, high = 144 / 180 * np.pi, size=1)
    alpha = np.random.uniform(low = -2*np.pi, high = 2*np.pi, size=1)
    height = np.random.uniform(low=0, high=100, size=1)
    
    params = np.array([kappa, beta, theta, phi, alpha, height])
   
    return np.squeeze(params)

def azimElevCoord(azim, elev, data):
    corz = np.cos(elev)
    xs = np.zeros([elev.shape[0]*azim.shape[0],4])
    n=0
    
    for k in np.arange(corz.shape[0]):
        cory = np.sin(elev[k])*np.sin(azim)
        corx = np.sin(elev[k])*np.cos(azim)
        for i in np.arange(corx.shape[0]):
            xs[n,0] = corx[i]
            xs[n,1] = cory[i]
            xs[n,2] = corz[k]
            xs[n,3] = data[k,i]
            n+=1

    return xs

def grid3d(gridsize = 200):

    u = np.linspace(0, 2 * np.pi, gridsize)
    v = np.linspace(0, np.pi, gridsize)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
#     keys = list()
#     points = list()
#     for i in range(gridsize):
#         for j in range(gridsize):
#             points.append([x[i, j], y[i, j], z[i, j]])
#             keys.append((i, j))
#     points = np.array(points)

#     value_for_color = np.ones(gridsize)
#     colors = np.empty((gridsize, gridsize), dtype=tuple)
#     for i, j in keys:
#         colors[i, j] = (1.0, 1.0, 1.0, 1.0)
        
    return x, y, z#, points, colors

def fitKent(data, xyz):
    '''
    X in radians
    Returns (height, xmean, conc)
    '''    
              #kappa, beta, theta, phi, alpha, height
    limits = ((0  , -0.5, -np.pi/4, -np.pi, -2*np.pi, 0),
              (100,  0.5,  3*np.pi/2,  np.pi,  2*np.pi, 100))
    
    params = kentRandStart()
    error_function = lambda p: np.ravel(kentdist(*p, xyz) - data)
    results = optimize.least_squares(error_function, params, bounds=limits)
    
    if results.success:
        try:
            p = results.x
            jacobian = results.jac
            cov = np.linalg.inv(jacobian.T.dot(jacobian))
#             cov = jacobian.T.dot(jacobian)
            var = np.sqrt(np.diagonal(cov))
            fun = results.fun
        except Exception as e:
            print(e)
            p = results.x * np.nan
            var = p
            fun = np.nan
    else:
        print(results.message)
        p = results.x * np.nan
        var = p
        fun = np.nan
    
    return p, var, fun


def aic_leastsquare(residuals, params):
    if not np.isnan(residuals).any():
        return residuals.size * np.log(np.var(residuals)) + 2*len(params)

def bic_leastsquare(residuals, params):
    if not np.isnan(residuals).any():
        return residuals.size * np.log(np.var(residuals)) + len(params)*np.log(residuals.size)


'''
# Example code on usage: 

from mpl_toolkits.mplot3d import Axes3D

data = np.array([[ 7.,  2.,  2.,  3.,  1.,  2.,  3.,  8.,  3., 15., 15., 14., 17., 13., 10.,  6.,  2.],
                 [ 6.,  7.,  7.,  3.,  6.,  9., 10.,  8.,  7., 20., 14., 17., 14., 18.,  9.,  7., 12.]])

azim = -1*np.array([-144, -126, -108, -90, -72, -54, -36, -18, 0, 18, 36, 54, 72, 90, 108, 126, 144])
elev = 90-np.array([0,40])
azim = azim*np.pi/180
elev = elev*np.pi/180

xs = azimElevCoord(azim, elev, data)
xyz = xs[:,:3]

interp = np.arange(azim[0],azim[-1],.1)
#fit the data niter times (random starts)
niter = 50

for n in np.arange(niter):
    p, var, residual = fitKent(xs[:,3], xyz)
    fitdist = kentdist(*p, xyz)
    if n == 0:
        param_store = np.zeros((niter, len(p)))
        resid_store = np.zeros(niter)
        var_store = np.zeros((niter, len(p)))
        fun_store = np.zeros(niter)
    resid_store[n] = np.sum(np.abs(residual))
    param_store[n] = p
    var_store[n] = var    
    fun_store[n] = aic_leastsquare(residual, p)

index = np.nanargmin(fun_store)
p = param_store[index]        
var = var_store[index]

param_labels = ['kappa', 'beta', 'theta', 'phi', 'alpha', 'height']#, 'base']
fig, axs = plt.subplots(4,2)
for pa, param in enumerate(param_labels):
    if param in ['theta', 'phi']:
        print(param, '{0} +/- {1}'.format(np.round(p[pa]*180/np.pi, 4), np.round(var[pa]*180/np.pi, 4)))
    else:
        print(param, p[pa])
    axs[pa%4][pa//4].hist(param_store[:,pa], bins=niter//2)
    axs[pa%4][pa//4].vlines(p[pa], 0, 10, 'red')
    axs[pa%4][pa//4].set_ylabel(param)

up = fitUniform(data.reshape(data.size), np.arange(data.size))
ufit = uniform(*up, np.arange(data.size))

uaic = aic_leastsquare(ufit.reshape(data.shape)-data, up)

index = np.nanargmin(fun_store)

axs[2][1].scatter(np.arange(niter), fun_store)
axs[2][1].scatter(index, fun_store[index],c='red')
axs[2][1].hlines(uaic, 0, 50)
axs[2][1].set_xlabel('iteration')
axs[2][1].set_ylabel('residual')

index = np.nanargmin(resid_store)

axs[3][1].scatter(np.arange(niter), resid_store)
axs[3][1].scatter(index, resid_store[index],c='red')
axs[3][1].set_xlabel('iteration')
axs[3][1].set_ylabel('residual')
plt.tight_layout()
plt.show()

#graph the data
fitdist = kentdist(*p, xyz).reshape(data.shape)

maxval = np.max(data)
minval=0

print(minval, maxval)
plt.imshow(data, vmin=minval, vmax=maxval)
plt.ylim(-0.5,4.5)
plt.yticks(np.arange(len(elev))[::2], np.round(elev[::2]*180/np.pi))
plt.xticks(np.arange(len(azim))[::4], np.round(azim[::4]*180/np.pi))
plt.show()

plt.imshow(fitdist, vmin=minval, vmax=maxval)
plt.ylim(-0.5,4.5)
plt.yticks(np.arange(len(elev))[::2], np.round(elev[::2]*180/np.pi))
plt.xticks(np.arange(len(azim))[::4], np.round(azim[::4]*180/np.pi))
plt.show()

kentres = chiSquaredTest(data.reshape(data.size), fitdist.reshape(fitdist.size), p)

'''
