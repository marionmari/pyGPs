import os,sys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
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
    plotDigit(exampleDigit7_1, 'This is a 7.')

    exampleDigit7_2 = x[129]   # another sample digit 2
    exampleDigit7_2 = np.reshape(exampleDigit7_2,(16,16))
    plotDigit(exampleDigit7_2, 'This is another 7.')

    # true class 1 	
    exampleDigitBad = x[70]    # digit that predicts wrong for rbf
    exampleDigitBad = np.reshape(exampleDigitBad,(16,16))
    plotDigit(exampleDigitBad, 'This digit is an example where the rbf kernel predicts the wrong class (2). Diffusion kernel predicts correctly due to graph information!')


    # form a 2-nearest neighbour graph 
    A = form_knn_graph(x,2)

    # use diffusion kernel to get precomputed matrix
    Matrix = diffKernel(A)
    N = Matrix.shape[0]

    # cross validation for RBF (no graph structure is used)
    num_folds = 10
    ACC = np.zeros(num_folds)
    i = 0;
    print '======= RBF =======' 
    for indice_train, indice_test in valid.k_fold_indice(N, K=num_folds):
	
        # compute kernel matrix          
        M1,M2 = form_kernel_matrix(Matrix, indice_train, indice_test)
	 
        # start Gaussian process
        model = gp.GPC()
        k = cov.RBF()
        model.setPrior(kernel=k)

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
        ACC[i] = valid.ACC(predictive_class, y_test)	
	#print np.hstack((np.array(indice_test, ndmin=2).T,y_test, predictive_class))
	print 'fold', i+1, ' accuracy: ' , ACC[i]
	i+=1;
	
    print 'mean accuracy: ', np.mean(ACC)
    print 'std accuracy: ', np.std(ACC)

    # cross validation for Diffusion kernel 
    # using indices because we need training/test indices for the precomputed kernel matrix
    num_folds = 10
    ACC = np.zeros(num_folds)
    i = 0;
    print '======= DIFF =======' 
    for indice_train, indice_test in valid.k_fold_indice(N, K=num_folds):
	
        # compute kernel matrix          
        M1,M2 = form_kernel_matrix(Matrix, indice_train, indice_test)
	 
        # start Gaussian process
        model = gp.GPC()
        k = cov.Pre(M1,M2)
        #k = cov.Pre(M1,M2) + cov.RBF()
        model.setPrior(kernel=k)
	
        # split training and test data
        y_train = y[indice_train,:]
        x_test = x[indice_test,:]
        y_test = y[indice_test,:] 

        # if you only use precomputed kernel matrix, there is no training data needed,
        # but you still need to specify x_train (due to general structure of pyGPs)
        # e.g. you can use the following:
        n = len(indice_train)
        x_train = np.zeros((n,1))

        # if you use combination of precomputed matrix and other kernel function,
        # this problem does not exsit
        # you can pass traning data in the normal way

        # gp
        model.train(x_train, y_train)
        model.predict(x_test)
        
        # evaluation 
        predictive_class = np.sign(model.ym)
        ACC[i] = valid.ACC(predictive_class, y_test)	
	#print np.hstack((np.array(indice_test, ndmin=2).T,y_test, predictive_class))
	print 'fold', i+1, ' accuracy: ' , ACC[i]
	i+=1;

    print 'mean accuracy: ', np.mean(ACC)
    print 'std accuracy: ', np.std(ACC)



