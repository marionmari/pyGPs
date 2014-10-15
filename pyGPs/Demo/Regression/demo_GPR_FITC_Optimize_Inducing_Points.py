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
u = np.asarray(random.sample(x,num_u))

# Currently only random initialization (should add cluster based and others...)

# Number of information pivots
num_info_pivots =  5

# Number of discrete swaps per pass
num_discrete_swaps_per_pass = 2

# and specify inducing point when seting prior
k = pyGPs.cov.RBF()
model.setPrior(kernel=k, inducing_points=u)

# The rest is analogous to what we have done before
model.setData(x, y)
model.fit()
print "Negative log marginal liklihood before optimization:", round(model.nlZ,3)
model.optimize()
print "Negative log marginal liklihood optimized:", round(model.nlZ,3)

# Prediction
ymu, ys2, fmu, fs2, lp = model.predict(z)
# Again, plot() is a toy method for 1-d data
model.plot()


print '--------------------END OF DEMO-----------------------'




