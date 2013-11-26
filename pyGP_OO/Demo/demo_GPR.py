from pyGP_OO.Core import *
import numpy as np

# export PYTHONPATH=$PYTHONPATH:/.../pyGP_OO/

# This demo will not only introduce GP regression model,
# but provides a gernel insight of our tourbox.

# You may want to read it before reading other models.
# current possible models are:
#     gp.GPR          -> Regression 
#     gp.GPC          -> Classification
#     gp.GPR_FITC     -> Sparse GP Regression 
#     gp.GPC_FITC     -> Sparse GP Classification
#     (gp.GPMC)       -> Muli-class Classification
#     (gp.GPMC_FITC)  -> Sparse Muli-class Classification


print ''
print '---------------------GPR DEMO-------------------------'

#----------------------------------------------------------------------
# Load demo data (generated from Gaussians)
#----------------------------------------------------------------------
demoData = np.load('data_for_demo/regression_data.npz')   
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data




#----------------------------------------------------------------------
# A five-line toy example
#----------------------------------------------------------------------
model = gp.GPR()             # model 
model.fit(x, y)              # fit default model (mean zero & rbf kernel) with data
model.train(x, y)            # optimize hyperparamters (default optimizer: single run minimize)
model.predict(z)             # predict test cases
model.plot()                 # and plot result




#----------------------------------------------------------------------
# Now lets do another example to get more insight to the toolbox
#----------------------------------------------------------------------

model = gp.GPR()            # start from a new model 

# Specify non-default mean and covariance functions 
# @SEE doc_kernel_mean for documentation of all kernels/means
m = mean.Linear( D=x.shape[1] ) + mean.Const()   
k = cov.RBF()
model.withPrior(mean=m, kernel=k) 


# Add traning data to model explictly,
# saves passing them each time when using fit() or train().
model.withData(x, y)
model.plotData_1d()


# Specify optimization method (single run "Minimize" by default)
# @SEE doc_optimization for documentation of optimization methods
model.setOptimizer("Minimize", num_restarts=30)


# Instead of fit(), which only fits data using given hyperparameters,
# train() will optimize hyperparamters based on marginal likelihood
model.train()


# There are several propertys you can get from the model
# For example:
#   model._neg_log_marginal_likelihood_
#   model._neg_log_marginal_likelihood_gradient_.cov
#   model._neg_log_marginal_likelihood_gradient_.lik
#   model._neg_log_marginal_likelihood_gradient_.mean
#   model._posterior_.sW
#   model._posterior_.alpha
#   model._posterior_.L
#   model.covfunc.hyp
#   model.meanfunc.hyp
#   model.likfunc.hyp
print 'Optimized negative log marginal liklihood:', round(model._neg_log_marginal_likelihood_,3)


# Predict test data
# output mean(ymu)/variance(ys2), latent mean(fmu)/variance(fs2), and log predictive probabilities(lp)
ymu, ys2, fmu, fs2, lp = model.predict(z)


# Set range of axis for plotting 
# NOTE: plot() is a toy method only for 1-d data
model.plot(axisvals=[-1.9, 1.9, -0.9, 3.9])




#----------------------------------------------------------------------
# A bit more things you can do
#----------------------------------------------------------------------

# [For all model] Speed up prediction time if you know posterior in advance
post = model._posterior_    # already known before
ymu, ys2, fmu, fs2, lp = model.predict_with_posterior( post,z )
# ...other than model.predict(z) 


# [Only for Regresstion] Specify noise of data (sigma=0.1 by default)
# You don't need it if you optimize it later anyway
model.withNoise( log_sigma=np.log(0.1) )

print '--------------------END OF DEMO-----------------------'


