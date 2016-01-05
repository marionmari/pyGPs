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
from pyGPs.Validation import valid
import numpy as np
from scipy.io import loadmat


# To have a gerneral idea,
# you may want to read demo_GPR, demo_kernel and demo_optimization first!
# Here, the focus is on multi-class classification.

print('')
print('---------------------GPMC DEMO-----------------------')

# GPMC is NOT based on multi-class Laplace Approximation.
# It works as a one vs. one classification wrapper

# i.e. GPMC trains GPC model for each combinations of two classes
# and uses voting scheme over all results to determine the final class

# It only returns the predictive class with highest rating, 
# but no other values (such as variance) returned. 

# Lets see a practical example to classify 10(0~9) hand-writen digits,
# using USPS digits dataset.


#----------------------------------------------------------------------
# Load USPS digits dataset
#----------------------------------------------------------------------
data = loadmat('usps_resampled.mat')
x = data['train_patterns'].T   # train patterns
y = data['train_labels'].T     # train labels
xs = data['test_patterns'].T   # test patterns
ys = data['test_labels'].T     # test labels   

# To be used in GPMC, we need to change label to integer from 0 to n(number of classes)
# here class value should be 0,1,...,9.
y = np.argmax(y, axis=1)
y = np.reshape(y, (y.shape[0],1))

ys = np.argmax(ys, axis=1)
ys = np.reshape(ys, (ys.shape[0],1))

# To save some time for demo, 
# lets reduce the number of training and testing patterns
x  = x[:100,:]
y  = y[:100,:]
xs = xs[:20,:]
ys = ys[:20,:]


#----------------------------------------------------------------------
# GPMC example
#----------------------------------------------------------------------

# State model with 10 classes
model = pyGPs.GPMC(10)

# Set data to model
model.setData(x,y)

# optimize default GPC model (see demo_GPC) for each binary classification problem, 
# and decide label for test patterns of hand-writen digits
# prdictive_vote[i,j] is the probability of being class j for test pattern i
predictive_vote = model.optimizeAndPredict(xs)

predictive_class = np.argmax(predictive_vote, axis=1)
predictive_class = np.reshape(predictive_class, (predictive_class.shape[0],1))

# Accuracy of recognized digit
acc = valid.ACC(predictive_class, ys)
print("Accuracy of recognizing hand-writen digits:", round(acc,2))


#----------------------------------------------------------------------
# A bit more things you can do
#----------------------------------------------------------------------
# Just like we did for GP classification
# You can use specify the setting for all binary classificiation problem by:
m = pyGPs.mean.Zero()
k = pyGPs.cov.RBF()
model.setPrior(mean=m,kernel=k)
model.useInference("Laplace")

# Beside optimizeAndPredict(xs),
# there is also an option to predict without optimization
# model.fitAndPredict(xs)

print('--------------------END OF DEMO-----------------------')

