#================================================================================
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [marthaler at ge dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]
#
#    This file is part of pyGP_OO.
#    The software package is released under the BSD 2-Clause (FreeBSD) License.
#
#    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 30/09/2013
#================================================================================


#------------------------------------------------------
# IN THIS DEMO: 
# How to use pyGP_OO in a cross-validation case...
# ...for data from e.g UCI.
#------------------------------------------------------
import os,sys
import numpy as np
from pyGP_OO.Valid import valid
from pyGP_OO.Pre import pre
from pyGP_OO.Core import *

#------------------------------------------------------
# Step 1: 
# load UCI dataset
# in this demo: ionosphere dataset
#------------------------------------------------------

file = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]),'../Data/ionosphere.data.txt'))
x, y = pre.load_data(file)



#------------------------------------------------------
# Step 2: More preprocessing depends on your data format
# here we deal with label in 'ionosphere.data.txt'
# "b" stands for "-1" and "g" stands for "+1"
#------------------------------------------------------
n,D = x.shape
for i in xrange(n):
    if y[i,0][0] == 'g':
        y[i,0] = 1
    else:
        y[i,0] = -1
y = np.int8(y)



#------------------------------------------------------
# Step 3: cross validation
# 10-fold validation
#------------------------------------------------------
K = 10
measure_ACC = []
measure_RMSE = []
for x_train, x_test, y_train, y_test in valid.k_fold_validation(x,y,K):
    '''
    IMPORTANT: 
    Since inf method will store some value in the last call...
    ..(e.g. last_alpha) so that can save computation if called again
    But here we are training different data for each k fold, 
    so last_alpha will get messed up, thus we need a new inf object everytime
    
    Hence I recommand to specify cov, mean, inf and lik functions here...
    ...if you specified these functions outside for loop, 
    then you need to know which values are stored in your specified inf method,
    and clear these values in the end of every iteration(e.g. i.last_alpha=None)
    '''
    k = cov.covSEiso([-1,0])
    m = mean.meanZero()
    l = lik.likErf()
    i = inf.infLaplace()
    o = opt.Minimize()

    '''
    If you know the hyperparameters to evaluate on,
    you can directly call gp.predict without training,
    which will be greatly faster
    '''
    gp.train(i,m,k,l,x_train,y_train,o)
    out = gp.predict(i,m,k,l,x_train,y_train,x_test,None)

    '''
    IMPORTANT: 
    When using GPC, the sign of out[0] is the predictive class 
    e.g. -0.7 means it belongs to class '-1'. 
    The far away the value from 0, the more confident the predition is.

    If you only care about class distribution in evaluation,
    write one more line as follows:

    pred_class = np.sign(out[0])

    Here I gave an example of evaluation on MSRE and Accuracy,
    see other evaluation measures in valid.py
    or define your own favourite measure
    '''
    pred_class = np.sign(out[0])
    acc = valid.ACC(pred_class, y_test)
    msre = valid.RMSE(pred_class, y_test)
    measure_ACC.append(acc)
    measure_RMSE.append(msre)

print 'average accuracy: ', np.mean(measure_ACC)
print 'average root-mean-square error: ', np.mean(measure_RMSE)





