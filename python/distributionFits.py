
'''
Common distributions that are used to fit the data
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import stats
from scipy import special


# Uniform distribution
def uniform(height):
    return lambda x: height*(x/x)


def uniformParams(data, X):
    height = np.mean(data)
    return height


def fitUniform(data, X):
    '''
    Returns (height, xmean, xsd)
    '''
    params =  uniformParams(data, X)
    error_function = lambda p: np.ravel(uniform(*p)(X) - data)
    p, success = optimize.leastsq(error_function, params)
    
    return p


#Gaussian distribution
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


#VonMises distribution
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


#Kent distribution
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


def kent(xyz, height, beta, kappa, gamma1, gamma2, gamma3, base):
    kent_dist = height * np.exp(-kappa) * np.exp(kappa * np.dot(xyz, gamma1) + 
            beta * kappa * (np.dot(xyz, gamma2)**2 - np.dot(xyz, gamma3)**2)) - base 
    return np.squeeze(kent_dist)


def kentdist(kappa, beta, theta, phi, alpha, height, base, xyz):
    alpha = 45

    theta_degree = 45 
    theta = theta_degree*np.pi/180

    phi_degree = 45
    phi = (90+phi_degree)*np.pi/180

    units = sphericalUnit(theta, phi)
    gamma1 = units[:, 0]

    gamma2 = rodrot(units[:, 1], units[:, 0], alpha)
    gamma3 = rodrot(units[:, 2], units[:, 0], alpha)

    gamma1 = np.transpose(gamma1[None,None,:], [1, 2, 0])
    gamma2 = np.transpose(gamma2[None,None,:], [1, 2, 0])
    gamma3 = np.transpose(gamma3[None,None,:], [1, 2, 0])

    return kent(xyz, height, beta, kappa, gamma1, gamma2, gamma3, base)


def kentRandStart():
    kappa = np.random.uniform(low=0, high=100, size=1)
    beta = np.random.uniform(low=-0.5, high=0.5, size=1)
    theta = np.random.uniform(low=0, high=np.pi/2, size=1)
    phi =  np.random.uniform(low = -144 / 180 * np.pi, high = 144 / 180 * np.pi, size=1)
    alpha = np.random.uniform(low = -2*np.pi, high = 2*np.pi, size=1)
    height = np.random.uniform(low=0, high=100, size=1)
    base = np.random.uniform(low=0, high=100, size=1)
    
    return kappa, beta, theta, phi, alpha, height, base


def fitKent(data, xyz):
    '''
    X in radians
    Returns (height, xmean, conc)
    '''
    params = kentRandStart()
    error_function = lambda p: np.ravel(kentdist(*p, xyz) - data)
    p, success = optimize.leastsq(error_function, params)
    
    return p


def azimElevCoord(azim, elev, data):
    corz = np.cos(elev)
    xs = np.zeros([elev.shape[0]*azim.shape[0],4])
    n=0
    
    for k in np.arange(corz.shape[0]):
        corx = np.sin(elev[k])*np.sin(azim)
        cory = np.sin(elev[k])*np.cos(azim)
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
    keys = list()
    points = list()
    for i in range(gridsize):
        for j in range(gridsize):
            points.append([x[i, j], y[i, j], z[i, j]])
            keys.append((i, j))
    points = np.array(points)

    value_for_color = np.ones(gridsize)
    colors = np.empty((gridsize, gridsize), dtype=tuple)
    for i, j in keys:
        colors[i, j] = (1.0, 1.0, 1.0, 1.0)
        
    return x, y, z, points, colors


def chiSquaredTest(data, modelfit, plot=True):
    
    normdata = np.sum(data)*modelfit/np.sum(modelfit)
    
    chiresults = stats.chisquare(data, normdata)
    
    print(chiresults)
    if plot:
        try:
            chibins = np.arange(0,int(chiresults.statistic*2),0.1)
        except:
            chibins = np.arange(0,100,0.1)
            
        fig, axs = plt.subplots(1,2)
        axs[0].title.set_text('Chi2 test')
        axs[0].set_ylabel('probability')
        axs[0].set_xlabel('Chi2 statistic')

        axs[0].plot(chibins, stats.chi2.pdf(chibins, df=len(data)-1))
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
        plt.show()

    return chiresults

