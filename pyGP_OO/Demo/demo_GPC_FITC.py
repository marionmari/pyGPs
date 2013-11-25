from pyGP_OO.Core import *
import numpy as np

# To have a gerneral idea, 
# you may want to read demo_GPR, demo_kernel and demo_optimization first!
# Here only focus on the difference of models.

print ''
print '-------------------GPC_FITC DEMO----------------------'

#----------------------------------------------------------------------
# Load demo data (generated from Gaussians)
#----------------------------------------------------------------------
demoData = np.load('data_for_demo/classification_data.npz')
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data
n = z.shape[0]               # number of test points

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

# Start from a new model 
model = gp.GPC_FITC()            

# Specify inducing points 
u1,u2 = np.meshgrid(np.linspace(-2,2,5),np.linspace(-2,2,5))
u = np.array(zip(np.reshape(u2,(np.prod(u2.shape),)),np.reshape(u1,(np.prod(u1.shape),)))) 

# For FITC model, you must specify prior explicitly 
# No default value since you need to specify inducing points 
m = mean.Zero()
k = cov.RBFard(log_ell_list=[0.05,0.17], log_sigma=1.)
model.withPrior(mean=m, kernel=k, inducing_points=u) 

# The rest is analogous to GPR
model.withData(x, y)
model.fit()
print "Negative log marginal liklihood before:", round(model._neg_log_marginal_likelihood_,3)
model.train()
print "Negative log marginal liklihood optimized:", round(model._neg_log_marginal_likelihood_,3)

# Prediction
ymu, ys2, fmu, fs2, lp = model.predict(z, ys=np.ones((n,1)))
# Again, plot() is a toy method for 2-d data
model.plot(x1,x2,t1,t2)



#----------------------------------------------------------------------
# A bit more things you can do
#----------------------------------------------------------------------

# Similar to GPC, GPC_FTIC uses EP FICT approximation by default,
# you can change to FITC Laplace Approximation by:
model.useLaplace_FITC()

print '--------------------END OF DEMO-----------------------'




