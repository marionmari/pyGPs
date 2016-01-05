from builtins import range
import numpy as np 
from scipy.io import loadmat
import os,sys

def load_binary(D1,D2,reduce=False):
    path = os.path.realpath(__file__)
    file_path = os.path.abspath(os.path.join(path,'../usps_resampled.mat'))
    data = loadmat(file_path)
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

if __name__ == '__main__':
    load_binary(1,2)