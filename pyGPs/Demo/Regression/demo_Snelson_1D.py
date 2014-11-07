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
import random

print ''
print '-------------------GPR_FITC Inducing Input Optimization DEMO----------------------'

#----------------------------------------------------------------------
# Load demo data (Snelson 1D data)
#----------------------------------------------------------------------
demoData = np.load('Snelson_1D_data.npz')
x = demoData['xtrain']       # training data
y = demoData['ytrain']       # training target
z = demoData['xstar']        # test data
print '------------------------------------------------------'
print "Example : optimize inducing points"

# Start from a new model
model = pyGPs.GPR_FITC()

# Initialize inducing points
# Pick a random subset of the training inputs.
num_u = 10
# Get num_u random indices of the training set
inducing_indices = random.sample(xrange(x.shape[0]),num_u)

u = x[inducing_indices]

k = pyGPs.cov.RBF()
model.setPrior(kernel=k, inducing_points=u)

# The rest is analogous to what we have done before
model.setData(x, y)

model.fit()
print "Negative log marginal liklihood before optimization:", model.nlZ
model.optimizeHyperparameters() # optimize hyperparameters with this inducing set
nlml = model.nlZ
print "Negative log marginal liklihood optimized:", nlml

k = pyGPs.cov.RBF()
model.setPrior(kernel=k, inducing_points=u)
model.optimizeInducingSet(num_u,'random')
nlml = model.nlZ
print "Negative log marginal liklihood optimized (random initialization) (inducing):", nlml

k = pyGPs.cov.RBF()
model.setPrior(kernel=k, inducing_points=u)
model.optimizeInducingSet(num_u,'cluster')
nlml = model.nlZ
print "Negative log marginal liklihood optimized (cluster initialization) (inducing):", nlml

# Prediction
ymu, ys2, fmu, fs2, lp = model.predict(z)
# Again, plot() is a toy method for 1-d data
model.plot()


print '--------------------END OF DEMO-----------------------'
