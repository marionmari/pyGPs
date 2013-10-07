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
import matplotlib.cm as cm
import matplotlib.mlab as mlab

#-----------------------------------------------------------------
# NOTICE
# For more detail and overal usage, see demoRegression.
# This demo only shows the differences when using GP classification.
#-----------------------------------------------------------------

print '\n------------------pyGP_OO classification DEMO----------------------'

PLOT = True

## LOAD DATA
print '...LOADING DATA...'
demoData = np.load('../Data/classification_data.npz')
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data
n = z.shape[0]               # number of test points

## DATA only needed in plotting (see build.py for more details)
if PLOT:  
    x1 = demoData['x1'] 
    x2 = demoData['x2']                    
    t1 = demoData['t1'] 
    t2 = demoData['t2'] 
    p1 = demoData['p1']  	# prior for class 1 (with label -1)
    p2 = demoData['p2']  	# prior for class 2 (with label +1)

## PLOT data 
if PLOT:
    fig = plt.figure()
    plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
    plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
    pc = plt.contour(t1, t2, np.reshape(p2/(p1+p2), (t1.shape[0],t1.shape[1]) ))
    fig.colorbar(pc)
    plt.grid()
    plt.axis([-4, 4, -4, 4])
    plt.show()


##--------------------------------------------------------------##
## Example 1:   prediction					##	
##--------------------------------------------------------------##
print '\n...Example 1: GP prediction...'

#-----------------------------------------------------------------
# Step 1:   SPECIFY combinations of cov, mean, inf and lik functions
#-----------------------------------------------------------------
k = cov.covSEard([0.05,0.17,1.21])
m =  mean.meanConst([0]) 
l = lik.likErf()
#i = inf.infLaplace()	# use Laplace approximation
i = inf.infEP()		# use expectation propagation

#-----------------------------------------------------------------
# Step 2:  ANALYZE GP (optional)
#-----------------------------------------------------------------
# get nlZ and dnlZ 
out = gp.analyze(i,m,k,l,x,y,True)
print "nlz =", out[0]
# print 'dnlZ.cov', out[1].cov
# print 'dnlZ.mean', out[1].mean
# print 'dnlZ.lik', out[1].lik

#-----------------------------------------------------------------
# Step 3:   GP PREDICTION
#-----------------------------------------------------------------
out = gp.predict(i,m,k,l,x,y,z,np.ones((n,1)))
ym = out[0]	# predictive mean
ys2 = out[1]	# predictive variance
fm = out[2]	# predictive latent mean
fs2 = out[3]	# predictive latent variance
lp = out[4]	# log predcitive probabilities

if PLOT:
    fig = plt.figure()
    plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
    plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
    pc = plt.contour(t1, t2, np.reshape(np.exp(lp), (t1.shape[0],t1.shape[1]) ))
    fig.colorbar(pc)
    plt.grid()
    plt.axis([-4, 4, -4, 4])
    plt.show()



##--------------------------------------------------------------##
## Example 2: optimization and prediction			##
## (Note: this example just shows functionality.)	
##--------------------------------------------------------------##
print '\n...Example 2: optimization and prediction...'

#-----------------------------------------------------------------
# Step 1:   SPECIFY combinations of cov, mean, inf and lik functions
#-----------------------------------------------------------------
k1 = cov.covSEard([0.0,0.0,0.1])
k2 = cov.covPoly([2,1,1])
k = k1*k2*6

m =  mean.meanConst([0]) + mean.meanLinear([1.,2.])
l = lik.likErf()
#i = inf.infLaplace()	# use Laplace approximation
i = inf.infEP()		# use expectation propagation


#-----------------------------------------------------------------
# Step 2:   SPECIFY optimization method
#-----------------------------------------------------------------
o = opt.Minimize()   # minimize by Carl Rasmussen
#o = opt.CG()        # conjugent gradient
#o = opt.BFGS()      # quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)
#o = opt.SCG()       # scaled conjugent gradient (faster than CG) 

#-----------------------------------------------------------------
# Step 3:  ANALYZE GP (optional)
#-----------------------------------------------------------------
# get nlZ and dnlZ 
out = gp.analyze(i,m,k,l,x,y,True)
print "nlz (no training) =", out[0]
# print 'dnlZ.cov', out[1].cov
# print 'dnlZ.mean', out[1].mean
# print 'dnlZ.lik', out[1].lik

#-----------------------------------------------------------------
# Step 4:   optimization/GP training 
#-----------------------------------------------------------------
out = gp.train(i,m,k,l,x,y,o)
print "nlz (optimal) =", out

#-----------------------------------------------------------------
# Step 5:   GP PREDICTION
#-----------------------------------------------------------------
out = gp.predict(i,m,k,l,x,y,z,np.ones((n,1)))
ym = out[0]	# predictive mean
ys2 = out[1]	# predictive variance
fm = out[2]	# predictive latent mean
fs2 = out[3]	# predictive latent variance
lp = out[4]	# log predcitive probabilities

if PLOT:
    fig = plt.figure()
    plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
    plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
    pc = plt.contour(t1, t2, np.reshape(np.exp(lp), (t1.shape[0],t1.shape[1]) ))
    fig.colorbar(pc)
    plt.grid()
    plt.axis([-4, 4, -4, 4])
    plt.show()



##--------------------------------------------------------------##
## More things you can do: 					##
## SPARSE GP							##
##--------------------------------------------------------------##
print '\n...SPARSE GP: FITC optimization and prediction...'
# SPECIFY inducing points
u1,u2 = np.meshgrid(np.linspace(-2,2,5),np.linspace(-2,2,5))
u = np.array(zip(np.reshape(u2,(np.prod(u2.shape),)),np.reshape(u1,(np.prod(u1.shape),)))) 

# SPECIFY FITC covariance functions
k = cov.covSEard([0.05,0.17,1.21]).fitc(u)

# SPECIFY FICT inference method
i = inf.infFITC_EP()
#i = inf.infFITC_Laplace()

# SPECIFY combinations of mean and lik functions (same as STANDARD GP)
m = mean.meanLinear([1.,2.]) + mean.meanConst([1.53])
l = lik.likErf()

# SPECIFY optimization method (same as STANDARD GP; see demoRegression.py Example 3 for details)
conf = pyGP_OO.Optimization.conf.random_init_conf(m,k,l)
conf.num_restarts = 5
o = opt.Minimize(conf)

# GET nlz and dnlz
out = gp.analyze(i,m,k,l,x,y,True)
print "[fitc] nlz =", out[0]
# print 'dnlZ.cov', out[1].cov
# print 'dnlZ.mean', out[1].mean
# print 'dnlZ.lik', out[1].lik

# OPTIMIZATION
nlZ_trained = gp.train(i,m,k,l,x,y,o)
print '[fitc] optimal nlz =', nlZ_trained

# PREDICTION
out = gp.predict(i,m,k,l,x,y,z,np.ones((n,1)))

if PLOT:
    fig = plt.figure()
    plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
    plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
    plt.plot(u[:,0],u[:,1],'ko', markersize=12)
    pc = plt.contour(t1, t2, np.reshape(np.exp(lp), (t1.shape[0],t1.shape[1]) ))
    fig.colorbar(pc)
    plt.grid()
    plt.axis([-4, 4, -4, 4])
    plt.show()        


print '------------------END OF DEMO----------------------'



