from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
#================================================================================
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [dan dot marthaler at gmail dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]
#
#    This file is part of pyGP_PR.
#    The software package is released under the BSD 2-Clause (FreeBSD) License.
#
#    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 18/02/2014
#================================================================================

import pyGPs
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    infile = 'housing.txt'
    data = np.genfromtxt(infile)

    DN, DD = data.shape
    N = 25
    # Get all data (exclude the 4th column which is binary) except the last 50 points for training
    x  = np.concatenate((data[:-N,:4],data[:-N,5:-1]),axis=1)
    x = old_div((x - np.mean(x,axis=0)),(np.std(x,axis=0)+1.e-16))
    # The function we will perform regression on:  Median Value of owner occupied homes
    y  = np.reshape(data[:-N,-1],(len(data[:-N,-1]),1))
    y = old_div((y-np.mean(y)),(np.std(y)+1.e-16))
    # Test on the last 50 points
    xs  = np.concatenate((data[-N:,:4],data[-N:,5:-1]),axis=1)
    xs = old_div((xs - np.mean(xs,axis=0)),(np.std(xs,axis=0)+1.e-16))
    ys = np.reshape(data[-N:,-1],(N,1))
    ys = old_div((ys-np.mean(ys)),(np.std(ys)+1.e-16))
    N,D = x.shape
    
    model = pyGPs.GPR()
    model.getPosterior(x, y)
    print('Initial negative log marginal likelihood = ', round(model.nlZ,3))
    
    # train and predict
    from time import clock
    t0 = clock()
    model.optimize(x,y)
    t1 = clock()
    ym, ys2, fm, fs2, lp = model.predict(xs)
    xa  = np.concatenate((data[:,:4],data[:,5:-1]),axis=1)
    xa = old_div((xa - np.mean(xa,axis=0)),(np.std(xa,axis=0)+1.e-16))
    ya, ys2a, fma, fs2a, lpa = model.predict(xa)

    print('Time to optimize = ', t1-t0)
    print('Optimized mean = ', model.meanfunc.hyp)
    print('Optimized covariance = ', model.covfunc.hyp)
    print('Optimized liklihood = ', model.likfunc.hyp)
    print('Final negative log marginal likelihood = ', round(model.nlZ,3))

    #HousingPlotter(range(len(y)),y,range(len(ym)),ym,ys2,range(len(y),len(y)+len(ys)),ys)
    xm = np.array(list(range(len(y),len(y)+ym.shape[0])))
    ym = np.reshape(ym,(ym.shape[0],))
    zm = np.reshape(ys2,(ym.shape[0],))

    plt.plot(ya,'g')
    plt.fill_between(xm, ym + 1.*np.sqrt(zm), ym - 1.*np.sqrt(zm), facecolor=[0.,1.0,0.0,0.9],linewidths=0.0)
    plt.fill_between(xm, ym + 2.*np.sqrt(zm), ym - 2.*np.sqrt(zm), facecolor=[0.,1.0,0.0,0.7],linewidths=0.0)
    plt.fill_between(xm, ym + 3.*np.sqrt(zm), ym - 3.*np.sqrt(zm), facecolor=[0.,1.0,0.0,0.5],linewidths=0.0)

    plt.plot(y,'r.',linewidth = 3.0, markersize = 5.0)
    plt.plot(xm,ym[-N:], 'bx', linewidth = 3.0, markersize = 5.0)
    plt.grid()
    plt.xlabel('Index')
    plt.ylabel('Median Home Values (normalized)')
    plt.axis([0.,510.,-3.5,3.5])
    plt.show()
