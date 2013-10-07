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
# For more detail and overal usages, see demoRegression
# This demo only shows the differences when using gp classification.
#-----------------------------------------------------------------



#-----------------------------------------------------------------
# initialze input data
#-----------------------------------------------------------------
PLOT = True

## LOAD data
demoData = np.load('../Data/classification_data.npz')
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data
n = z.shape[0]           # number of test points

## DATA only needed in plotting
if PLOT:  
    x1 = demoData['x1'] 
    x2 = demoData['x2']                    
    t1 = demoData['t1'] 
    t2 = demoData['t2'] 
    p1 = demoData['p1']  
    p2 = demoData['p2']  

## PLOT data 
if PLOT:
    print '...LOADING DATA...'
    fig = plt.figure()
    plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
    plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
    pc = plt.contour(t1, t2, np.reshape(p2/(p1+p2), (t1.shape[0],t1.shape[1]) ))
    fig.colorbar(pc)
    plt.grid()
    plt.axis([-4, 4, -4, 4])
    plt.show()



#-----------------------------------------------------------------
# step 1:
# specify combinations of cov, mean, inf and lik functions
#-----------------------------------------------------------------
k1 = cov.covSEard([0.05,0.17,1.21])
k2 = cov.covSEiso([-1,0])
k3 = cov.covPoly([2,1,1])
k = k1*k3*6
m = mean.meanLinear([1.,2.]) + mean.meanConst([1.53])
l = lik.likErf()
#i = inf.infLaplace()
i = inf.infEP()


#-----------------------------------------------------------------
# step 2 (optional):
# specify optimization methods
#-----------------------------------------------------------------
o = opt.Minimize()
#o = opt.CG()

#-----------------------------------------------------------------
# step 3:
# optimization, training and prediction
#-----------------------------------------------------------------
# analyze nlZ and dnlZ 
out = gp.analyze(i,m,k,l,x,y,True)
print "nlz=", out[0]
# print 'dnlZ.cov', out[1].cov
# print 'dnlZ.mean', out[1].mean
# print 'dnlZ.lik', out[1].lik

# training (find optimal hyperparameters)
out = gp.train(i,m,k,l,x,y,o)
print "optimal nlz=", out

# predict after optimization
out = gp.predict(i,m,k,l,x,y,z,np.ones((n,1)))
a = out[0]; b = out[1]; c = out[2]; d = out[3]; lp = out[4]
#print a
if PLOT:
    print '...Example 1 Training and Prediction...'
    fig = plt.figure()
    plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
    plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
    pc = plt.contour(t1, t2, np.reshape(np.exp(lp), (t1.shape[0],t1.shape[1]) ))
    fig.colorbar(pc)
    plt.grid()
    plt.axis([-4, 4, -4, 4])
    plt.show()



#-----------------------------------------------------------------
# More things you can do: SPARSE GP
#-----------------------------------------------------------------
# specify inducing points
u1,u2 = np.meshgrid(np.linspace(-2,2,5),np.linspace(-2,2,5))
u = np.array(zip(np.reshape(u2,(np.prod(u2.shape),)),np.reshape(u1,(np.prod(u1.shape),)))) 

# specify FITC covariance functions
k = cov.covSEard([0.05,0.17,1.21]).fitc(u)

# specify FICT inference method
i = inf.infFITC_EP()
#i = inf.infFITC_Laplace()

# The rest usage is the same as STANDARD GP
# if you are not sure about sth, see demoRegressition for detail
# Here we just give one example:
m = mean.meanLinear([1.,2.]) + mean.meanConst([1.53])
l = lik.likErf()

# optimization method
conf = pyGP_OO.Optimization.conf.random_init_conf(m,k,l)
conf.max_trails = 10
o = opt.Minimize(conf)

# analyze nlz and dnlz
out = gp.analyze(i,m,k,l,x,y,True)
print "[fitc] nlz=", out[0]
# print 'dnlZ.cov', out[1].cov
# print 'dnlZ.mean', out[1].mean
# print 'dnlZ.lik', out[1].lik

# training
nlZ_trained = gp.train(i,m,k,l,x,y,o)
print '[fitc] optimal nlZ=', nlZ_trained

# predict and plot
out = gp.predict(i,m,k,l,x,y,z,np.ones((n,1)))

if PLOT:
    print '...Example 2 FITC Training and Prediction...'
    fig = plt.figure()
    plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
    plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
    plt.plot(u[:,0],u[:,1],'ko', markersize=12)
    pc = plt.contour(t1, t2, np.reshape(np.exp(lp), (t1.shape[0],t1.shape[1]) ))
    fig.colorbar(pc)
    plt.grid()
    plt.axis([-4, 4, -4, 4])
    plt.show()        



#-----------------------------------------------------------------
# end of demo
#-----------------------------------------------------------------

print '------------------END OF DEMO----------------------'



