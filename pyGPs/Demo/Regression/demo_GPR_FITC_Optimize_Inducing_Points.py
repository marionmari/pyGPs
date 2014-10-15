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
unchecked_indices = inducing_indices[:]
checked_indices = []
remaining_pool = list( set(range(x.shape[0])) - set(unchecked_indices) )

u = x[inducing_indices]

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
print inducing_indices
model.optimizeInducingSet()
'''model.fit()
print "Negative log marginal liklihood before optimization:", model.nlZ
model.optimize() # optimize hyperparameters with this inducing set
nlml = model.nlZ
print "Negative log marginal liklihood optimized:", nlml
'''
# Loop over inducing points to find the best set
'''while unchecked_indices:

    # pick a random inducing variable (that hasn't been checked) to replace
    j = random.choice(unchecked_indices)
    # Delete this element from the list
    inducing_indices.remove(j) # removes the element = j
    unchecked_indices.remove(j) # removes the element = j
    # Add a new element from the remaining pool
    i = random.choice(remaining_pool)  # Need to choose which one to check better than random
    remaining_pool.remove(i)
    inducing_indices.append(i)

    #set the prior with this new inducing set
    model.setPrior(kernel=k, inducing_points=u)

    model.optimize() # optimize hyperparameters with this inducing set
    nlml_new = model.nlZ

    print "Negative log marginal liklihood optimized (new set):", nlml_new
    if nlml_new > nlml:
        # put j back in
        inducing_points.remove(i)
        inducing_points.append(j)
'''
# Prediction
#ymu, ys2, fmu, fs2, lp = model.predict(z)
# Again, plot() is a toy method for 1-d data
#model.plot()


print '--------------------END OF DEMO-----------------------'




