#================================================================================
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [marthaler at ge dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]
#
#    This file is part of pyGPs.
#    The software package is released under the BSD 2-Clause (FreeBSD) License.
#
#    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 18/02/2014
#================================================================================


import numpy as np

def k_fold_validation(x, y, K=10, randomise=False):
    """
    Generates K (training, validation) pairs from the items in X.

    The validation iterables are a partition of X, and each validation
    iterable is of length len(X)/K. Each training iterable is the
    complement (within X) of the validation iterable, and so each training
    iterable is of length (K-1)*len(X)/K.
    """
    # whether needed to shuffle data first
    if randomise:
        data = np.append(x, y, axis=1)
        np.random.shuffle(data)
        x = data[:,:-1]
        y = data[:,-1]

    n, D = x.shape
    assert n > K
    for k in xrange(K):
        x_train = np.array([e for i, e in enumerate(x) if i % K != k])
        x_test = np.array([e for i, e in enumerate(x) if i % K == k])
        y_train = np.array([e for i, e in enumerate(y) if i % K != k])
        y_test = np.array([e for i, e in enumerate(y) if i % K == k])
        yield x_train, x_test, y_train, y_test


def k_fold_index(n, K=10):
    '''
    Similar to k_fold_validation,
    but only return the indice instead of data

    n is number of instances of data
    K is number of folder 
    '''
    for k in xrange(K):
        indice_train = []
        indice_test = []
        for i in xrange(n):
            if i % K == k:
                indice_test.append(i)
            else:
                indice_train.append(i)
        yield indice_train,indice_test


def RMSE(predict,target):
    '''
    Root mean squared error
    '''
    error = predict - target
    return np.sqrt(np.mean(error**2))

def ACC(predict,target):
    '''
    Classification accuracy
    '''
    n,D = target.shape
    count = 0.
    for i in xrange(n):
        if predict[i,0] == target[i,0]:
            count += 1
        #else:
        #    print i 
            
    return count/n


def Prec(predict,target):
    '''
    Precision for class +1
    '''
    n,D = target.shape
    count_1 = 0.
    count_2 = 0.
    for i in xrange(n):
        if predict[i,0] == 1:
            count_1 += 1
            if target[i,0] == 1:
                count_2 += 1
    return count_2 / count_1


def Recall(predict,target):
    '''
    Recall for class +1
    '''
    n,D = target.shape
    count_1 = 0.
    count_2 = 0.
    for i in xrange(n):   
        if target[i,0] == 1:
            count_1 += 1
            if predict[i,0] == 1:
                count_2 += 1
    return count_2 / count_1


def NLPD(y, MU, S2):
    '''
    Calculate evaluation measure NLPD in transformed observation space.
       
       INPUT   y     observed targets
               MU    vector of predictions/predicted means
               S2    vector of 'self' variances
               
       OUTPUT  nlpd  Negative Log Predictive Density.
    '''
    nlpd = 0.5*log(2*math.pi*S2) + 0.5*((y-MU)**2)/S2 
    nlpd = np.mean(nlpd)
    return nlpd
    




