from __future__ import division
from builtins import range
from past.utils import old_div
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


import numpy as np

def k_fold_validation(x, y, K=10, randomise=False):
    '''
    Generates K (training, validation) pairs from the items in X.
    The validation iterables are a partition of X, and each validation
    iterable is of length len(X)/K. Each training iterable is the
    complement (within X) of the validation iterable, and so each training
    iterable is of length (K-1)*len(X)/K.

    :param x: training data
    :param y: training targets
    :param K: number of folds
    :param randomise: boolean flag. Shuffle data first if it is true.s
    '''
    # whether needed to shuffle data first
    if randomise:
        data = np.append(x, y, axis=1)
        np.random.shuffle(data)
        x = data[:,:-1]
        y = data[:,-1]

    n, D = x.shape
    assert n > K
    for k in range(K):
        x_train = np.array([e for i, e in enumerate(x) if i % K != k])
        x_test = np.array([e for i, e in enumerate(x) if i % K == k])
        y_train = np.array([e for i, e in enumerate(y) if i % K != k])
        y_test = np.array([e for i, e in enumerate(y) if i % K == k])
        yield x_train, x_test, y_train, y_test


def k_fold_index(n, K=10):
    '''
    Similar to k_fold_validation,
    but only yields indice of folds instead of data in each iteration

    :param n: size of data (number of instances)
    :param K: number of folds

    '''
    for k in range(K):
        indice_train = []
        indice_test = []
        for i in range(n):
            if i % K == k:
                indice_test.append(i)
            else:
                indice_train.append(i)
        yield indice_train,indice_test


def RMSE(predict,target):
    '''
    Root mean squared error

    :param predict: vector of predicted means
    :param target: vector of true means
    :return: root mean squared error
    '''
    error = predict - target
    return np.sqrt(np.mean(error**2))

def ACC(predict,target):
    '''
    Classification accuracy

    :param predict: vector of predicted labels(+/- 1)
    :param target: vector of true labels
    :return: accuracy
    '''
    n,D = target.shape
    count = 0.
    for i in range(n):
        if predict[i,0] == target[i,0]:
            count += 1
    return old_div(count,n)


def Prec(predict,target):
    '''
    Precision for class +1

    :param predict: vector of predicted labels(+/- 1)
    :param target: vector of true labels
    :return: precision
    '''
    n,D = target.shape
    count_1 = 0.
    count_2 = 0.
    for i in range(n):
        if predict[i,0] == 1:
            count_1 += 1
            if target[i,0] == 1:
                count_2 += 1
    return old_div(count_2, count_1)


def Recall(predict,target):
    '''
    Recall for class +1

    :param predict: vector of predicted labels(+/- 1)
    :param target: vector of true labels
    :return: recall
    '''
    n,D = target.shape
    count_1 = 0.
    count_2 = 0.
    for i in range(n):   
        if target[i,0] == 1:
            count_1 += 1
            if predict[i,0] == 1:
                count_2 += 1
    return old_div(count_2, count_1)


def NLPD(y, MU, S2):
    '''
    Calculate evaluation measure NLPD in transformed observation space.
    
    :param y: observed targets
    :param MU: vector of predictions/predicted means
    :param S2: vector of 'self' variances
    :return: Negative Log Predictive Density.
    '''
    nlpd = 0.5*log(2*math.pi*S2) + 0.5*((y-MU)**2)/S2 
    nlpd = np.mean(nlpd)
    return nlpd
    




