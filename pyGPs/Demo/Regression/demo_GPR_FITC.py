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

# To have a gerneral idea,
# you may want to read demo_GPR, demo_kernel and demo_optimization first!
# Here, the focus is on the difference of FITC model.

print ''
print '-------------------GPR_FITC DEMO----------------------'

#----------------------------------------------------------------------
# Load demo data (generated from Gaussians)
#----------------------------------------------------------------------
demoData = np.load('regression_data.npz')

x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data


#----------------------------------------------------------------------
# Sparse GP regression (FITC) example
#----------------------------------------------------------------------
print "Example 1: default inducing points"

# Start from a new model
model = pyGPs.GPR_FITC()

# Notice if you want to use default inducing points:
# You MUST call setData(x,y) FIRST!
# The default inducing points are a grid (hypercube in higher dimension), where
# each dimension has 5 values in equidistant steps between min and max value of the input data by default.
model.setData(x, y)

# To set value per dimension use:
# model.setData(x, y, value_per_axis=10)

model.optimizeHyperparameters()
print "Negative log marginal liklihood optimized:", round(model.nlZ,3)

# Prediction
model.predict(z)
# Again, plot() is a toy method for 1-d data
model.plot()



print '------------------------------------------------------'
print "Example 2: user-defined inducing points"

# Start from a new model
model = pyGPs.GPR_FITC()

# You can define inducing points yourself.
# You can pick some points by hand
u = np.array([[-1], [-0.8], [-0.5], [0.3],[1.]])

# or equally-spaced inducing points
num_u = np.fix(x.shape[0]/2)
u = np.linspace(-1.3,1.3,num_u).T
u = np.reshape(u,(num_u,1))

# and specify inducing point when seting prior
m = pyGPs.mean.Linear( D=x.shape[1] ) + pyGPs.mean.Const()
k = pyGPs.cov.RBF()
model.setPrior(mean=m, kernel=k, inducing_points=u)

# The rest is analogous to what we have done before
model.setData(x, y)
model.fit()
print "Negative log marginal liklihood before optimization:", round(model.nlZ,3)
model.optimizeHyperparameters()
print "Negative log marginal liklihood optimized:", round(model.nlZ,3)

# Prediction
ymu, ys2, fmu, fs2, lp = model.predict(z)
# Again, plot() is a toy method for 1-d data
model.plot()


print '------------------------------------------------------'
print "Example 3: optimize inducing points (with no initial guess)"

# Start from a new model
model = pyGPs.GPR_FITC()
model.setData(x, y)
model.setPrior(mean=m, kernel=k)
model.u = None

# You can define the number of inducing points
num_u = int(np.fix(x.shape[0]/2))

m = pyGPs.mean.Linear( D=x.shape[1] ) + pyGPs.mean.Const()
k = pyGPs.cov.RBF()

model.optimizeInducingSet(num_u)
nlml = model.nlZ
print "Negative log marginal liklihood optimized (inducing):", nlml

# Prediction
ymu, ys2, fmu, fs2, lp = model.predict(z)
# Again, plot() is a toy method for 1-d data
model.plot()

print '------------------------------------------------------'
print "Example 4: optimized inducing points (with initial guess)"

# Start from a new model
model = pyGPs.GPR_FITC()

# You can choose inducing points yourself, but they must be a subset of the training data
num_u = int(np.fix(x.shape[0]/2))
u = x[random.sample(xrange(x.shape[0]),num_u)]

# and specify inducing point when seting prior
m = pyGPs.mean.Linear( D=x.shape[1] ) + pyGPs.mean.Const()
k = pyGPs.cov.RBF()
model.setPrior(mean=m, kernel=k, inducing_points=u)

# The rest is analogous to what we have done before
model.setData(x, y)
model.fit()
print "Negative log marginal liklihood before optimization:", round(model.nlZ,3)

model.optimizeHyperparameters()
print "Negative log marginal liklihood optimized:", round(model.nlZ,3)

print "About to call optimizeInducingSet\n"
model.optimizeInducingSet()
nlml = model.nlZ
print "Negative log marginal liklihood optimized (inducing):", nlml

# Prediction
ymu, ys2, fmu, fs2, lp = model.predict(z)
# Again, plot() is a toy method for 1-d data
model.plot()


print '--------------------END OF DEMO-----------------------'




