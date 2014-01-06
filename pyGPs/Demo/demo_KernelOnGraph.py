import os,sys
import numpy as np
from scipy.io import loadmat
from pyGPs.Core import *
from pyGPs.GraphStuff.graph_util import *
from pyGPs.GraphStuff.kernels_on_graph import *
from pyGPs.Valid import valid


def load_binary(D1,D2,reduce=False):
    # path = os.path.realpath(__file__)
    # file_path = os.path.abspath(os.path.join(path,'../usps_resampled.mat'))
    data = loadmat('data_for_demo/usps_resampled.mat')
    x = data['train_patterns'].T   # train patterns
    y = data['train_labels'].T     # train_labels
    xx = data['test_patterns'].T   # test patterns
    yy = data['test_labels'].T     # test labels
    D1_list = []
    D2_list = []
    n,D = x.shape
    for i in xrange(n):
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


if __name__ == "__main__":
    # load small reduced dataset with 2 classes
    # digit 1 for +1 and digit 2 for -1
    x,y = load_binary(1,7,reduce=True)

    # form a 2-nearest neighbour graph 
    A = form_knn_graph(x,2)

    # use diffusion kernel to get precomputed matrix
    Matrix = diffKernel(A)
    N = Matrix.shape[0]

    # cross validation 
    # using indices because we need training/test indice to precompute kernel matrix
    for indice_train, indice_test in valid.k_fold_indice(N, K=10):

        # compute kernel matrix
        n = len(indice_train)             
        M1,M2 = form_kernel_matrix(Matrix, indice_train, indice_test)

        # start Gaussian process
        model = gp.GPC()
        k = cov.Pre(M1,M2) + cov.RBF()

        # if you only use precomputed kernel matrix, there is no training data,
        # but you still need to specify x (due to generality of pyGPs)
        # you can create the following:
        # x_train = np.zeros((n,1))

        # if you use combination of precomputed matrix and other kernel function,
        # this problem does not exsit
        # you can use pyGPs in the normal way

        # split training and test data
        x_train = x[indice_train,:]
        y_train = y[indice_train,:]
        x_test = x[indice_test,:]
        y_test = y[indice_test,:]
        
        # gp
        model.train(x_train, y_train)
        model.predict(x_test)

        # evaluation
        predictive_class = np.sign(model.ym)
        acc = valid.ACC(predictive_class, y_test)
        print 'Accuracy: ' , acc


