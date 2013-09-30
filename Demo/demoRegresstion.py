#================================================================================
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [marthaler at ge dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]
#
#    This file is part of pyGP_OO.
#    The software package is released under the BSD 2-Clause (FreeBSD) License.
#
#    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 30/09/2013
#================================================================================


# export PYTHONPATH=$PYTHONPATH:/.../pyGPs/

import pyGP_OO
from pyGP_OO.Core import *
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------
# initialze input data
#-----------------------------------------------------------------
PLOT = True

# DATA
demoData = np.load('../Data/regression_data.npz')
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data

if PLOT:
    print '...LOADING DATA...'
    pyGP_OO.Visual.plot.datasetPlotter(x,y,[-1.9, 1.9, -0.9, 3.9])  



#-----------------------------------------------------------------
# Step 1:
# specify combinations of cov, mean, inf and lik functions
#-----------------------------------------------------------------

# In this demo I am focusing on illustring how to use composite functions
# therefore it might NOT be the optimal kernel choice for this data 
k1 = cov.covSEiso([-1,0])
k2 = cov.covPoly([1,1,2])
k3 = cov.ScaleOfKernel(k2,6)
# in pyGP_OO, you can directly use +,* to combine kernels
# and +,*,** to combine mean functions
k = k1*k3

# NOTICE:
# Here alternatively you can use the simpler form: k = k1*k2*6
# scalar 6 will stll be treated as a hyperparameter that will be trained
# i.e hyperparameter list is [-1,0,1,1,6]
# [-1,0] is hyperparameter of k1, 
# [1,1] is of k2(degree is a special parameter which can not be optimized. see cov.py)
# and [6] is the scalar

m = mean.meanZero()
l = lik.likGauss([np.log(0.1)])
i = inf.infExact()



#-----------------------------------------------------------------
# Step 2 (optional):
# specify optimization methods
#-----------------------------------------------------------------
conf = pyGP_OO.Optimization.conf.random_init_conf(m,k,l)

# you can specify number of trials during optimization
# this is the number of running optimization method 
# inside optimization for each trial, there will still be many iterations wrt. the chosen method
conf.max_trails = 20

# you can also specify the minimal value you want to achieve
# e.g. conf.min_threshold = 100

# and you can set range of hyperparameters 
# it is the range of initial guess of each trial for optimizaion methods
# during optimization, it may find optimal hyp out of this range
# if you do not set range, the default will be (-10,10) for each hyperparameter
conf.covRange = [(-10,10), (-10,10), (-10,10),(-10,10),(5,6)]
conf.likRange = [(0,1)]

o = opt.Minimize(conf)   # minimize by Carl Rasmussen
#o = opt.CG(conf)        # conjugent gradient
#o = opt.BFGS(conf)      # quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)
#o = opt.SCG(conf)       # scaled conjugent gradient (faster than CG) 

# You can also use optimization without configuration
# then you will run the oprimization for only one trial with random initial guess
# e.g. o = opt.Minimize()



#-----------------------------------------------------------------
# analyze nlZ and dnlZ(optional)
#-----------------------------------------------------------------
out = gp.analyze(i,m,k,l,x,y,True)
print 'nlZ=', out[0]
# print 'dnlZ.cov', out[1].cov
# print 'dnlZ.mean', out[1].mean
# print 'dnlZ.lik', out[1].lik



#-----------------------------------------------------------------
# predict without optimization (optional)
#-----------------------------------------------------------------
out = gp.predict(i,m,k,l,x,y,z)
ym  = out[0]
ys2 = out[1]
mm  = out[2]
s2  = out[3]
if PLOT:
    print '...Example 1 Prediction without Optimization...'
    pyGP_OO.Visual.plot.standardPlotter(z,ym,ys2,x,y,[-1.9, 1.9, -0.9, 3.9])



#-----------------------------------------------------------------
# Step 3: 
# training (find optimal hyperparameters)
#-----------------------------------------------------------------
nlZ_trained = gp.train(i,m,k,l,x,y,o)
print 'optimal nlZ=', nlZ_trained



#-----------------------------------------------------------------
# Step 4:
# predict after training
#-----------------------------------------------------------------
out = gp.predict(i,m,k,l,x,y,z)
ym  = out[0]
ys2 = out[1]
mm  = out[2]
s2  = out[3]
if PLOT:
    print '...Example 2 Training and Prediction...'
    pyGP_OO.Visual.plot.standardPlotter(z,ym,ys2,x,y,[-1.9, 1.9, -0.9, 3.9])



#-----------------------------------------------------------------
# More things you can do: SPARSE GP
#-----------------------------------------------------------------
print 'More: SPARSE GP'
# specify inducing points
n = x.shape[0]
num_u = np.fix(n/2)
u = np.linspace(-1.3,1.3,num_u).T
u  = np.reshape(u,(num_u,1))

# specify FITC covariance functions
k = cov.covSEiso([-1.6, -0.45]).fitc(u)
# specify FICT inference method
i = inf.infFITC_Exact()

# The rest way of calling gp is the same as STANDARD GP
# Here we just give one example:
m = mean.meanLinear([1.18]) + mean.meanConst([1.53])
l = lik.likGauss([-1.])
# specify optimization method
conf = pyGP_OO.Optimization.conf.random_init_conf(m,k,l)
conf.max_trails = 10
conf.covRange = [(-1,0), (-1,0)]
conf.likRange = [(-3.5,-3)]
conf.meanRange = [(1,2),(1,2)]
o = opt.Minimize(conf)
#o = opt.SCG(conf)

# get nlz
out = gp.analyze(i,m,k,l,x,y,True)
print "[fitc] nlz=", out[0]
#print "[fitc] dnlz.mean=", out[1].mean
#print "[fitc] dnlz.cov=", out[1].cov
#print "[fitc] dnlz.lik=", out[1].lik

# training
nlZ_trained = gp.train(i,m,k,l,x,y,o)
print '[fitc] optimal nlZ=', nlZ_trained
#print k.hyp
#print m.hyp
#print l.hyp


# prediction
out = gp.predict(i,m,k,l,x,y,z)
ymF = out[0]
y2F = out[1]
mF  = out[2]
s2F = out[3]

if PLOT:
    print '...Example 3: FITC Training and Prediction...'
    pyGP_OO.Visual.plot.fitcPlotter(u,z,ymF,y2F,x,y,[-1.9, 1.9, -0.9, 3.9])
    plt.show()      # show all figures now at the same time

#-----------------------------------------------------------------
# end of demo
#-----------------------------------------------------------------

print '------------------END OF DEMO----------------------'


