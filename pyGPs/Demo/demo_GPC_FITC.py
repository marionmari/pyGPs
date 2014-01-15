from pyGPs.Core import *
import numpy as np

# To have a gerneral idea, 
# you may want to read demo_GPR, demo_kernel and demo_optimization first!
# Here, the focus is on the difference of FITC classification.

print ''
print '-------------------GPC_FITC DEMO----------------------'

#----------------------------------------------------------------------
# Load demo data (generated from Gaussians)
#----------------------------------------------------------------------

# GPC_FITC target class are +1 and -1
demoData = np.load('data_for_demo/classification_data.npz')
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data

# only needed for 2-d contour plotting 
x1 = demoData['x1']          # x for class 1 (with label -1)
x2 = demoData['x2']          # x for class 2 (with label +1)     
t1 = demoData['t1']          # y for class 1 (with label -1)
t2 = demoData['t2']          # y for class 2 (with label +1)
p1 = demoData['p1']          # prior for class 1 (with label -1)
p2 = demoData['p2']          # prior for class 2 (with label +1)




#----------------------------------------------------------------------
# Sparse GP classification (FITC) example
#----------------------------------------------------------------------

print "Example 1: default inducing points"

# Start from a new model 
model = gp.GPC_FITC()            

# Notice if you want to use default inducing points:
# You MUST call setData(x,y) FIRST!
# The default inducing points is a grid(hypercube in higher dimension), where
# each dimension has 5 values in same step between min and max value of data by default.
model.setData(x, y)

# To set value per dimension use:
# model.setData(x, y, value_per_axis=10)

model.train()
print "Negative log marginal liklihood optimized:", round(model.nlZ,3)

# Prediction
n = z.shape[0]              
model.predict(z, ys=np.ones((n,1)))
# Again, plot() is a toy method for 2-d data
model.plot(x1,x2,t1,t2)



print '------------------------------------------------------'
print "Example 2: user-defined inducing points"

model = gp.GPC_FITC() 

# You can define inducing points yourself.
# u = np.array([])
u1,u2 = np.meshgrid(np.linspace(-2,2,5),np.linspace(-2,2,5))
u = np.array(zip(np.reshape(u2,(np.prod(u2.shape),)),np.reshape(u1,(np.prod(u1.shape),)))) 

# and specify inducing point when seting prior
m = mean.Zero()
k = cov.RBFard(log_ell_list=[0.05,0.17], log_sigma=1.)
model.setPrior(mean=m, kernel=k, inducing_points=u) 

# The rest is analogous to what we have done before.
model.setData(x, y)
model.fit()
print "Negative log marginal liklihood before optimization:", round(model.nlZ,3)
model.train()
print "Negative log marginal liklihood optimized:", round(model.nlZ,3)

# predict
n = z.shape[0]              
ymu, ys2, fmu, fs2, lp = model.predict(z, ys=np.ones((n,1)))
model.plot(x1,x2,t1,t2)



#----------------------------------------------------------------------
# A bit more things you can do
#----------------------------------------------------------------------

# Similar to GPC, GPC_FTIC uses EP FICT approximation by default,
# you can change to FITC Laplace Approximation by:
model.useLaplace_FITC()

print '--------------------END OF DEMO-----------------------'




