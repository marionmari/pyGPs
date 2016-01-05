from __future__ import print_function
from builtins import str
from builtins import range
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

# To have a gerneral idea,
# you may want to read demo_GPR, demo_kernel and demo_optimization first!

# This example shows the process of cross-validation using real dataset.

print('')
print('---------------Cross-Validation DEMO------------------')

#----------------------------------------------------------------------
# Load raw data (ionosphere dataset from UCI)
#----------------------------------------------------------------------
data_source = "ionosphere.data.txt"
x = []
y = []
with open(data_source) as f:
    for index,line in enumerate(f):
        feature = line.split(',')
        attr = feature[:-1]
        attr = [float(i) for i in attr]
        target = [feature[-1]]       
        x.append(attr)
        y.append(target)
x = np.array(x)
y = np.array(y)



#----------------------------------------------------------------------
# Data cleaning/preprocessing
#----------------------------------------------------------------------
# Here we deal with label in ionosphere data,
# change "b" to"-1", and "g" to "+1"
n,D = x.shape
for i in range(n):
    if y[i,0][0] == 'g':
        y[i,0] = 1
    else:
        y[i,0] = -1
y = np.int8(y)


#----------------------------------------------------------------------
# Cross Validation
#----------------------------------------------------------------------
K = 10             # number of fold
ACC = []           # accuracy 
RMSE = []          # root-mean-square error

cv_run = 0
for x_train, x_test, y_train, y_test in valid.k_fold_validation(x, y, K):
    print('Run:', cv_run)
    # This is a binary classification problem
    model = pyGPs.GPC()
    # Since no prior knowldege, leave everything default 
    model.optimize(x_train, y_train)
    # Predit 
    ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys=y_test)

    # ymu for classification is a continuous value over -1 to +1
    # If you want predicting result to either one of the classes, take a sign of ymu.
    ymu_class = np.sign(ymu)

    # Evluation
    acc = valid.ACC(ymu_class, y_test)
    print('   accuracy =', round(acc,2)) 
    rmse = valid.RMSE(ymu_class, y_test)
    print('   rmse =', round(rmse,2))
    ACC.append(acc)
    RMSE.append(rmse)

    # Toward next run
    cv_run += 1   

print('\nAccuracy: ', np.round(np.mean(ACC),2), '('+str(np.round(np.std(ACC),2))+')')
print('Root-Mean-Square Error: ', np.round(np.mean(RMSE),2))




#----------------------------------------------------------------------
# Evaluation measures
#----------------------------------------------------------------------
'''
We defined the following measures(@SEE pyGPs.Valid.valid):

    RMSE - Root mean squared error
    ACC - Classification accuracy
    Prec - Precision for class +1
    Recall - Recall for class +1
    NLPD - Negative log predictive density in transformed observation space
'''


print('--------------------END OF DEMO-----------------------')
