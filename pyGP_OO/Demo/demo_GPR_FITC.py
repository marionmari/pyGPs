from pyGP_OO.Core import *
import numpy as np

# To have a gerneral idea, 
# you may want to read demo_GPR, demo_kernel and demo_optimization first!
# Here only focus on the difference of models.

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

# Start from a new model 
model = gp.GPR_FITC()            

# Specify inducing points 
num_u = np.fix(x.shape[0]/2)
u = np.linspace(-1.3,1.3,num_u).T
u = np.reshape(u,(num_u,1))

# For FITC model, you must specify prior explicitly 
# No default value since you need to specify inducing points 
m = mean.Linear( D=x.shape[1] ) + mean.Const()  
k = cov.RBF()
model.withPrior(mean=m, kernel=k, inducing_points=u) 

# The rest is analogous to GPR
model.withData(x, y)
model.fit()
print "Negative log marginal liklihood before:", round(model._neg_log_marginal_likelihood_,3)
model.train()
print "Negative log marginal liklihood optimized:", round(model._neg_log_marginal_likelihood_,3)

# Prediction
ymu, ys2, fmu, fs2, lp = model.predict(z)
# Again, plot() is a toy method for 1-d data
model.plot()


print '--------------------END OF DEMO-----------------------'




