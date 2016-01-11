from __future__ import print_function
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

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pyGPs
from pyGPs.GraphExtensions import graphUtil, nodeKernels
from pyGPs.Validation import valid


def load_binary(D1,D2,reduce=False):
    data = loadmat('usps_resampled.mat')
    x = data['train_patterns'].T   # train patterns
    y = data['train_labels'].T     # train_labels
    xx = data['test_patterns'].T   # test patterns
    yy = data['test_labels'].T     # test labels
    D1_list = []
    D2_list = []
    n,D = x.shape
    for i in range(n):
        if y[i,D1] == 1:
            D1_list.append(i)
        elif y[i,D2] == 1:
            D2_list.append(i)
    if reduce == True:
        D1_list = D1_list[:100]
        D2_list = D2_list[:100]
    n1 = len(D1_list)
    n2 = len(D2_list)
    x_binary = np.concatenate((x[D1_list,:], x[D2_list,:]))
    y_binary = np.concatenate((np.ones((n1,1)),-np.ones((n2,1))))
    return x_binary,y_binary


def plotDigit(digit, title_str=''):
    fig, ax = plt.subplots()
    ax.imshow(digit, cmap=plt.cm.gray, interpolation='nearest')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.title(title_str)
    plt.show()


if __name__ == "__main__":
    # load small reduced dataset with 2 classes
    # digit 1 for +1 and digit 2 for -1
    x,y = load_binary(1,2,reduce=True)
  
    # plot example digit
    exampleDigit1_1 = x[38]    # sample digit 1
    exampleDigit1_1 = np.reshape(exampleDigit1_1,(16,16))
    plotDigit(exampleDigit1_1, 'This is a 1.')

    exampleDigit1_2 = x[81]    # another sample digit 1
    exampleDigit1_2 = np.reshape(exampleDigit1_2,(16,16))
    plotDigit(exampleDigit1_2, 'This is another 1.')

    exampleDigit7_1 = x[124]   # sample digit 2
    exampleDigit7_1 = np.reshape(exampleDigit7_1,(16,16))
    plotDigit(exampleDigit7_1, 'This is a 2.')

    exampleDigit7_2 = x[129]   # another sample digit 2
    exampleDigit7_2 = np.reshape(exampleDigit7_2,(16,16))
    plotDigit(exampleDigit7_2, 'This is another 2.')

    # true class 1     
    exampleDigitBad = x[70]    # digit that predicts wrong for rbf
    exampleDigitBad = np.reshape(exampleDigitBad,(16,16))
    plotDigit(exampleDigitBad, 'This digit is an example where the rbf kernel predicts the wrong class (2). \nDiffusion kernel predicts correctly due to graph information!')
    
    # true class 2     
    exampleDigitBad = x[190]    # digit that predicts wrong for diff
    exampleDigitBad = np.reshape(exampleDigitBad,(16,16))
    plotDigit(exampleDigitBad, 'This digit is an example where the diff kernel predicts the wrong class (1). \nrbf kernel, however, predicts correctly!')


    # form a 2-nearest neighbour graph 
    A = graphUtil.formKnnGraph(x,2)

    # use diffusion kernel to get precomputed matrix
    Matrix = nodeKernels.diffKernel(A)
    N = Matrix.shape[0]
    
    # set training and test set
    index_train = range(N)
    index_test = [38, 81, 124, 129, 70, 190]
    index_train =  np.setdiff1d(index_train, index_test)
        
    ## RBF kernel
    # initialize Gaussian process
    model = pyGPs.GPC()
    k = pyGPs.cov.RBF()
    model.setPrior(kernel=k)

    # split training and test data
    x_train = x[index_train,:]
    y_train = y[index_train,:]
    x_test = x[index_test,:]
    y_test = y[index_test,:]
    
    # gp
    model.optimize(x_train, y_train)
    model.predict(x_test)
    
    # evaluation 
    predictive_class_rbf = np.sign(model.ym)
    ACC_rbf = valid.ACC(predictive_class_rbf, y_test)    

    ## DIFFUSION Kernel
    # compute kernel matrix and initalize GP with precomputed kernel                  
    model = pyGPs.GPC()
    M1,M2 = graphUtil.formKernelMatrix(Matrix, index_train, index_test)
    k = pyGPs.cov.Pre(M1,M2)          
    model.setPrior(kernel=k)

    # if you only use precomputed kernel matrix, there is no training data needed,
    # but you still need to specify x_train (due to general structure of pyGPs)
    # e.g. you can use the following:
    n = len(index_train)
    x_train = np.zeros((n,1))

    # gp
    model.optimize(x_train, y_train)
    model.predict(x_test)
    
    # evaluation 
    predictive_class_diff = np.sign(model.ym)
    ACC_diff = valid.ACC(predictive_class_diff, y_test)
    
    ## SUM of DIFFUSION and RBF Kernel
    # compute kernel matrix and initalize GP with precomputed kernel                  
    model = pyGPs.gp.GPC()
    M1,M2 = graphUtil.formKernelMatrix(Matrix, index_train, index_test)
    k = pyGPs.cov.Pre(M1,M2) + pyGPs.cov.RBFunit()
    model.setPrior(kernel=k)

    # if you use combination of precomputed matrix and other kernel function,
    # you can pass traning data in the normal way: x_train = x[index_train,:]
    x_train = x[index_train,:]
    # gp
    model.optimize(x_train, y_train)
    model.predict(x_test)
    
    # evaluation 
    predictive_class_sum = np.sign(model.ym)
    ACC_sum = valid.ACC(predictive_class_sum, y_test)
    
    
    print(np.hstack((np.array(index_test, ndmin=2).T, y_test, predictive_class_rbf, predictive_class_diff, predictive_class_sum)))
    print('accuracy (RBF): ' , ACC_rbf)
    print('accuracy (DIFF): ' , ACC_diff)
    print('accuracy (SUM): ' , ACC_sum)


   

