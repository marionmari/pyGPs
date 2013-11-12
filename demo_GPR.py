from pyGP_OO.Core import *
import numpy as np

print ''
print '------------------pyGP regression DEMO----------------------'

#----------------------------------------------------------------------
# Load demo data (generated from gaussian)
#----------------------------------------------------------------------
demoData = np.load('pyGP_OO/Data/regression_data.npz')   
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data



#----------------------------------------------------------------------
# A four-line toy example
#----------------------------------------------------------------------
model = gp.GPR()             # state model 
model.fit(x, y)              # fit model with data
model.predict(z)             # predict test cases
model.plot()                 # and plot result



#----------------------------------------------------------------------
# Now lets do a more detailed example to get familiar with the toolbox
#----------------------------------------------------------------------

# By default, we are using mean zero and rbf kernel,
# You can specify other priors. 
# (If you will optimize anyway, just leave parameters to be default)
m = mean.Linear() + mean.Const()     
k = cov.rbf()
model.withPrior(mean=m, kernel=k) 

# By default, data is noisy with sigma=0.1,
# you can also set other variances.
# (If you will optimize anyway, this step is not needed)
model.withNoise( log_sigma=np.log(0.1) )

# You can also add data to model explictly,
# then you will save passing x,y every time you use fit() or train().
model.withData(x, y)
model.plotData()



# Instead of fit(), which only fits data using given hyperparameters,
# train() will optimize hyperparamters based on marginal likelihood
model.train()

# There are several property you can get from the model
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
print 'Optimized marginal liklihood:', model._neg_log_marginal_likelihood_


# Output mean/variance, latent mean/variance, and log predictive probabilities
ymu, ys2, fmu, fs2, lp = model.predict(z)

# You can specify range of axis for your plot 
model.plot(axisvals=[-1.9, 1.9, -0.9, 3.9])



"""
# TODO  add demo_kernel 
# @see demo_kernel  for all default settings
# @see demo_kernel  for how to use kernel(mean) compositions and how to set hyperparameters
# @add demo_kernel  for why using logarithm of ell and sigma (to ensure positive)

"""