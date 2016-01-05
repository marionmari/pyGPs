from __future__ import print_function
#================================================================================
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [dan dot marthaler at gmail dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]
#
#    This file is part of pyGPs.
#    The software package is released under the BSD 2-Clause (FreeBSD) License.
#
#    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 18/02/2014
#================================================================================

import pyGPs
import numpy as np

# This demo will not only introduce GP regression model,
# but provides a gerneral insight of our tourbox.

# You may want to read it before reading other models.
# current possible models are:
#     pyGPs.GPR          -> Regression
#     pyGPs.GPC          -> Classification
#     pyGPs.GPR_FITC     -> Sparse GP Regression
#     pyGPs.GPC_FITC     -> Sparse GP Classification
#     pyGPs.GPMC         -> Muli-class Classification



print('')
print('---------------------GPR DEMO-------------------------')

#----------------------------------------------------------------------
# Load demo data (generated from Gaussians)
#----------------------------------------------------------------------
demoData = np.load('regression_data.npz')
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data

#----------------------------------------------------------------------
# A five-line example
#----------------------------------------------------------------------
print('Basic Example')
model = pyGPs.GPR()          # model
print('Before Optimization')
model.setData(x,y)
model.predict(z)             # predict test cases (before optimization)
model.plot()                 # and plot result
model.optimize(x, y)         # optimize hyperparamters (default optimizer: single run minimize)
print('After Optimization')
model.predict(z)             # predict test cases
model.plot()                 # and plot result

#----------------------------------------------------------------------
# Now lets do another example to get more insight to the toolbox
#----------------------------------------------------------------------
print('More Advanced Example (using a non-zero mean and Matern7 kernel)')
model = pyGPs.GPR()           # start from a new model

# Specify non-default mean and covariance functions
# SEE doc_kernel_mean for documentation of all kernels/means
m = pyGPs.mean.Const() + pyGPs.mean.Linear()
k = pyGPs.cov.Matern(d=7) # Approximates RBF kernel
model.setPrior(mean=m, kernel=k)



# Specify optimization method (single run "Minimize" by default)
# @SEE doc_optimization for documentation of optimization methods
#model.setOptimizer("RTMinimize", num_restarts=30)
#model.setOptimizer("CG", num_restarts=30)
#model.setOptimizer("LBFGSB", num_restarts=30)

# Instead of getPosterior(), which only fits data using given hyperparameters,
# optimize() will optimize hyperparamters based on marginal likelihood
# the deafult mean will be adapted to the average value of the training labels.
# ..if you do not specify mean function by your own.
model.optimize(x, y)

# There are several properties you can get from the model
# For example:
#   model.nlZ
#   model.dnlZ.cov
#   model.dnlZ.lik
#   model.dnlZ.mean
#   model.posterior.sW
#   model.posterior.alpha
#   model.posterior.L
#   model.covfunc.hyp
#   model.meanfunc.hyp
#   model.likfunc.hyp
#   model.ym (predictive means)
#   model.ys2 (predictive variances)
#   model.fm (predictive latent means)
#   model.fs2 (predictive latent variances)
#   model.lp (log predictive probability)
print('Optimized negative log marginal likelihood:', round(model.nlZ,3))


# Predict test data
# output mean(ymu)/variance(ys2), latent mean(fmu)/variance(fs2), and log predictive probabilities(lp)
ym, ys2, fmu, fs2, lp = model.predict(z)


# Set range of axis for plotting
# NOTE: plot() is a toy method only for 1-d data
model.plot()
# model.plot(axisvals=[-1.9, 1.9, -0.9, 3.9]))


#----------------------------------------------------------------------
# A bit more things you can do
#----------------------------------------------------------------------

# [For all model] Speed up prediction time if you know posterior in advance
post = model.posterior    # already known before


ym, ys2, fmu, fs2, lp = model.predict_with_posterior(post,z)
# ...other than model.predict(z)


# [Only for Regresstion] Specify noise of data (sigma=0.1 by default)
# You don't need it if you optimize it later anyway
model.setNoise( log_sigma=np.log(0.1) )

print('--------------------END OF DEMO-----------------------')

