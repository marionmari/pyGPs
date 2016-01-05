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

# To have a gerneral idea,
# you may want to read demo_GPR, demo_kernel and demo_optimization first!
# Here, the focus is on the difference of classification model.

print('')
print('---------------------GPC DEMO-------------------------')

#----------------------------------------------------------------------
# Load demo data (generated from Gaussians)
#----------------------------------------------------------------------
# GPC target class are +1 and -1
demoData = np.load('classification_data.npz')
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data

# only needed for 2-d contour plotting 
x1 = demoData['x1']          # x for class 1 (with label -1)
x2 = demoData['x2']          # x for class 2 (with label +1)     
t1 = demoData['t1']          # y for class 1 (with label -1)
t2 = demoData['t2']          # y for class 2 (with label +1)
p1 = demoData['p1']          # prior for class 1 (with label -1)
p2 = demoData['p2']          # prior for class 2 (with label +1)




#----------------------------------------------------------------------
# First example -> state default values
#----------------------------------------------------------------------
print('Basic Example - Data')
model = pyGPs.GPC()          # binary classification (default inference method: EP)
model.plotData_2d(x1,x2,t1,t2,p1,p2)
model.getPosterior(x, y)     # fit default model (mean zero & rbf kernel) with data
model.optimize(x, y)         # optimize hyperparamters (default optimizer: single run minimize)
model.predict(z)             # predict test cases

print('Basic Example - Prediction')
model.plot(x1,x2,t1,t2)

#----------------------------------------------------------------------
# GP classification example
#----------------------------------------------------------------------
print('More Advanced Example')
# Start from a new model 
model = pyGPs.GPC()    

# Analogously to GPR
k = pyGPs.cov.RBFard(log_ell_list=[0.05,0.17], log_sigma=1.)
model.setPrior(kernel=k) 

model.getPosterior(x, y)
print("Negative log marginal liklihood before:", round(model.nlZ,3))
model.optimize(x, y)
print("Negative log marginal liklihood optimized:", round(model.nlZ,3))

# Prediction
n = z.shape[0]
ymu, ys2, fmu, fs2, lp = model.predict(z, ys=np.ones((n,1)))

# pyGPs.GPC.plot() is a toy method for 2-d data
# plot log probability distribution for class +1
model.plot(x1,x2,t1,t2)

print('--------------------END OF DEMO-----------------------')




