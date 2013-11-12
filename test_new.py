from pyGP_OO.Core import *
import numpy as np


print '\n------------------pyGP_OO regression DEMO----------------------'

PLOT = True

# LOAD DATA
demoData = np.load('pyGP_OO/Data/regression_data.npz')
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data


model = gp.GPR()
model.withData(x,y)

#m = mean.meanLinear([0.5]) + mean.meanConst([1.])
#k = cov.covPoly([np.log(0.25),np.log(1.0),1.0])
k = cov.covPoly([2,1,1]) 
model.withPrior(kernel=k)


model.fit()
print model._neg_log_marginal_likelihood_


model.predict(z)
model.plotPrediction(axisvals=[-1.9, 1.9, -0.9, 3.9])

model.train()
print model._neg_log_marginal_likelihood_

model.predict(z)
model.plotPrediction(axisvals=[-1.9, 1.9, -0.9, 3.9])

"""
print model._neg_log_marginal_likelihood_gradient_.cov
print model._neg_log_marginal_likelihood_gradient_.lik
print model._neg_log_marginal_likelihood_gradient_.mean

print model._posterior_.sW
print model._posterior_.alpha
print model._posterior_.L
"""
