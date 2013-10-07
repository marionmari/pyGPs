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


import pyGP_OO
from pyGP_OO.Core import *
import numpy as np
import matplotlib.pyplot as plt

print '------------------pyGP_OO DEMO----------------------'

PLOT = True

# LOAD DATA
print '...LOADING DATA...'
demoData = np.load('../Data/regression_data.npz')
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data

if PLOT:
    pyGP_OO.Visual.plot.datasetPlotter(x,y,[-1.9, 1.9, -0.9, 3.9])  


##--------------------------------------------------------------##
## Example 1: GP prediction w/o optimization			##	
##--------------------------------------------------------------##
print '\n...Example 1: prediction without optimization...'

#-----------------------------------------------------------------
# Step 1:   specify combinations of cov, mean, inf and lik functions
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
# [1,1] is of k2(degree is a special parameter which is not optimized. See cov.py)
# and [6] is the scalar

m = mean.meanZero()
l = lik.likGauss([np.log(0.1)])
i = inf.infExact()

#-----------------------------------------------------------------
# Step 2:  ANALYZE GP (optional)
#-----------------------------------------------------------------
# get nlZ (and dnlZ) 
# where nlZ 	= value of the negative log marginal likelihood
#   	dnlZ	= column vector of partial derivatives of the negative
#                 log marginal likelihood w.r.t. each hyperparameter
out = gp.analyze(i,m,k,l,x,y,True)
print 'nlZ =', out[0]
# print 'dnlZ.cov', out[1].cov
# print 'dnlZ.mean', out[1].mean
# print 'dnlZ.lik', out[1].lik



#-----------------------------------------------------------------
# Step 3:   GP PREDICTION w/o optimization
#-----------------------------------------------------------------
out = gp.predict(i,m,k,l,x,y,z)
ym  = out[0]
ys2 = out[1]
mm  = out[2]
s2  = out[3]
if PLOT:
    pyGP_OO.Visual.plot.standardPlotter(z,ym,ys2,x,y,[-1.9, 1.9, -0.9, 3.9])


##--------------------------------------------------------------##
## Example 2:  basic optimization and prediction  		##
##--------------------------------------------------------------##
print '\n...Example 2: basic optimization and prediction...'

#-----------------------------------------------------------------
# Step 1:   specify combinations of cov, mean, inf and lik functions
#-----------------------------------------------------------------
# (see Example 1)

#-----------------------------------------------------------------
# Step 2:  SPECIFY optimization method   
#-----------------------------------------------------------------
o = opt.Minimize()   # minimize by Carl Rasmussen
#o = opt.CG()        # conjugent gradient
#o = opt.BFGS()      # quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)
#o = opt.SCG()       # scaled conjugent gradient (faster than CG) 

#-----------------------------------------------------------------
# Step 3:   RUN optimization/GP training (find optimal hyperparameters; one random initialization)
#-----------------------------------------------------------------
nlZ_trained = gp.train(i,m,k,l,x,y,o)
print 'nlZ =', nlZ_trained

#-----------------------------------------------------------------
# Step 4:   GP PREDICTION
#-----------------------------------------------------------------
out = gp.predict(i,m,k,l,x,y,z)
ym  = out[0]
ys2 = out[1]
mm  = out[2]
s2  = out[3]
if PLOT:
    pyGP_OO.Visual.plot.standardPlotter(z,ym,ys2,x,y,[-1.9, 1.9, -0.9, 3.9])



##--------------------------------------------------------------##
## Example 3: optimization with restarts and prediction  	##
##--------------------------------------------------------------##
print '\n...Example 3: optimization with restarts and prediction...'

#-----------------------------------------------------------------
# Step 1:   specify combinations of cov, mean, inf and lik functions
#-----------------------------------------------------------------
# (see Example 1)


#-----------------------------------------------------------------
# Step 2:   SPECIFY optimization method (using iterative restarts)
#-----------------------------------------------------------------
conf = pyGP_OO.Optimization.conf.random_init_conf(m,k,l)

# you can specify number of restarts during optimization
# this is the number of runs of the optimization method 
# inside optimization for each trial, there will still be many iterations wrt. the chosen method
conf.num_restarts = 20

# SPECIFY the minimal value you want to achieve (optional)
#conf.min_threshold = 15

# SET range of hyperparameters 
# it is the range of initial guess of each trial for optimizaion
# during optimization, it may find optimal hyp out of this range
# if you do not set range, the default will be (-10,10) for each hyperparameter
conf.covRange = [(-10,10), (-10,10), (-10,10),(-10,10),(5,6)]
conf.likRange = [(0,1)]

# SPECIFY optimization method
o = opt.Minimize(conf)   # minimize by Carl Rasmussen
#o = opt.CG(conf)        # conjugent gradient
#o = opt.BFGS(conf)      # quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)
#o = opt.SCG(conf)       # scaled conjugent gradient (faster than CG) 


#-----------------------------------------------------------------
# Step 3:   RUN optimization/GP training (find optimal hyperparameters; use random restarts)
#-----------------------------------------------------------------
nlZ_trained = gp.train(i,m,k,l,x,y,o)
print 'optimal nlZ=', nlZ_trained


#-----------------------------------------------------------------
# Step 4:   GP PREDICTION
#-----------------------------------------------------------------
out = gp.predict(i,m,k,l,x,y,z)
ym  = out[0]
ys2 = out[1]
mm  = out[2]
s2  = out[3]
if PLOT:
    pyGP_OO.Visual.plot.standardPlotter(z,ym,ys2,x,y,[-1.9, 1.9, -0.9, 3.9])



##--------------------------------------------------------------##
## More things you can do: 					##
## GET GP posterior						##
##--------------------------------------------------------------##
print '\n...GP POSTERIOR: to get the GP posterior call gp.analyze()...'
# there are 2 ways to get the GP posterior:
post  = gp.analyze(i,m,k,l,x,y,True)[2]
# or
post  = gp.analyze(i,m,k,l,x,y,False)[1]

alpha = post.alpha 
L     = post.L
sW    = post.sW



##--------------------------------------------------------------##
## More things you can do: 					##
## SPARSE GP							##
##--------------------------------------------------------------##
print '\n...SPARSE GP: FITC optimization and prediction...'
# SPECIFY inducing points
n = x.shape[0]
num_u = np.fix(n/2)
u = np.linspace(-1.3,1.3,num_u).T
u  = np.reshape(u,(num_u,1))

# SPECIFY FITC covariance functions
k = cov.covSEiso([-1.6, -0.45]).fitc(u)

# SPECIFY FICT inference method
i = inf.infFITC_Exact()

# SPECIFY combinations of mean and lik functions (same as STANDARD GP)
m = mean.meanLinear([1.18]) + mean.meanConst([1.53])
l = lik.likGauss([-1.])

# SPECIFY optimization method (same as STANDARD GP)
conf = pyGP_OO.Optimization.conf.random_init_conf(m,k,l)
conf.num_restarts = 10
conf.covRange = [(-1,0), (-1,0)]
conf.likRange = [(-3.5,-3)]
conf.meanRange = [(1,2),(1,2)]
o = opt.Minimize(conf)

# GET nlz
out = gp.analyze(i,m,k,l,x,y,True)
print "[fitc] nlz=", out[0]
#print "[fitc] dnlz.mean=", out[1].mean
#print "[fitc] dnlz.cov=", out[1].cov
#print "[fitc] dnlz.lik=", out[1].lik

# OPTIMIZATION
nlZ_trained = gp.train(i,m,k,l,x,y,o)
print '[fitc] optimal nlZ=', nlZ_trained
#print k.hyp
#print m.hyp
#print l.hyp


# PREDICTION
out = gp.predict(i,m,k,l,x,y,z)
ymF = out[0]
y2F = out[1]
mF  = out[2]
s2F = out[3]

if PLOT:
    pyGP_OO.Visual.plot.fitcPlotter(u,z,ymF,y2F,x,y,[-1.9, 1.9, -0.9, 3.9])

#-----------------------------------------------------------------
# end of demo
#-----------------------------------------------------------------

print '------------------END OF DEMO----------------------'


