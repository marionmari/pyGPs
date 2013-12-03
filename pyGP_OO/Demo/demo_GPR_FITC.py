from pyGP_OO.Core import *
import numpy as np

# To have a gerneral idea, 
# you may want to read demo_GPR, demo_kernel and demo_optimization first!
# Here, the focus is on the difference of FITC model.

print ''
print '-------------------GPR_FITC DEMO----------------------'

#----------------------------------------------------------------------
# Load demo data (generated from Gaussians)
#----------------------------------------------------------------------
demoData = np.load('data_for_demo/regression_data.npz')   
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data




#----------------------------------------------------------------------
# Sparse GP regression (FITC) example
#----------------------------------------------------------------------

print "Example 1: deafult inducing points"

# Start from a new model 
model = gp.GPR_FITC()            

# Notice if you want to use default inducing points:
# You MUST call setData(x,y) FIRST!
# The default inducing points is a grid(hypercube in higher dimension), where
# each axis has 5 values in same step between min and max value of data in this dimension.
model.setData(x, y, value_per_axis=5)
model.train()
print "Negative log marginal liklihood optimized:", round(model._neg_log_marginal_likelihood_,3)

# Prediction             
model.predict(z)
# Again, plot() is a toy method for 1-d data
model.plot()



print '------------------------------------------------------'
print "Example 2: user-defined inducing points"

# Start from a new model 
model = gp.GPR_FITC()            

# You can define inducing points yourself. 
num_u = np.fix(x.shape[0]/2)
u = np.linspace(-1.3,1.3,num_u).T
u = np.reshape(u,(num_u,1))

# and specify inducing point when seting prior
m = mean.Linear( D=x.shape[1] ) + mean.Const()  
k = cov.RBF()
model.setPrior(mean=m, kernel=k, inducing_points=u) 

# The rest is analogous to what we have done before
model.setData(x, y)
model.fit()
print "Negative log marginal liklihood before:", round(model._neg_log_marginal_likelihood_,3)
model.train()
print "Negative log marginal liklihood optimized:", round(model._neg_log_marginal_likelihood_,3)

# Prediction
ymu, ys2, fmu, fs2, lp = model.predict(z)
# Again, plot() is a toy method for 1-d data
model.plot()


print '--------------------END OF DEMO-----------------------'




