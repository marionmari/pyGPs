from __future__ import division
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

import scipy.spatial as spspatial
import scipy.sparse as spsp
import numpy as np

def formKnnGraph(pc,k):
    '''
    Form a k-nearest-neighbour graph from data points

    :param pc: n by D data matrix
    :param k: number of neighbours for each node
    :return: adjacency matrix
    '''

    num_points = pc.shape[0]
   
    tree =  spspatial.KDTree(pc)    
    [dist, nearest_neighbours] = tree.query(pc, k+1)
    nearest_neighbours = nearest_neighbours[:,1:]       # no self edges
    weights = np.ones((dist.shape[0],dist.shape[1]-1))  # same weights for edges
    row_idx = np.kron(np.arange(0,num_points),np.ones((k,1))).T

    num_edges = row_idx.shape[0]*row_idx.shape[1]    

    A = spsp.coo_matrix((weights.reshape(num_edges),  (row_idx.reshape(num_edges),nearest_neighbours.reshape(num_edges))), shape=(num_points,num_points))   
    A = np.array(A.todense())
    A = np.maximum(A,A.T)   # make symmetric
    return A



def formKernelMatrix(M, indice_train, indice_test):
    '''
    Format precomputed kernel matrix into two matrix,
    which fit the structure to be used in cov.Pre() in pyGP

    :param M: n by n precomputed kernel matrix
    :param indice_train: list of indice of training examples
    :param indice_test: list of indice of test examples
        
    :return: M1 is a train+1 by test matrix, 
    where the last row is the diagonal of test-test covariance.
    and M2 is a train by train matrix.
    '''
    train_test = M[indice_train,:][:,indice_test]
    test_test = M[indice_test,:][:,indice_test]
    dia = np.diag(test_test)
    test_dia = np.reshape(dia, (1,dia.shape[0]))
    M1 = np.concatenate((train_test,test_dia))
    M2 = M[indice_train,:][:,indice_train]
    return M1,M2


def normalizeKernel(K):
    '''
    Normalize the given kernel matrix. 
    Each entry[i,j] is normalized by square root of entry[i,i] * entry[j,j].
    (i.e. compute the correlation matrix from covariance matrix).

    :param K: n by D kernel matrix(covariance matrix)
    :return: n by D normalized kernel matrix(correlation matrix)
    '''
    Kdiag = np.atleast_2d(np.diag(K))
    normalizers = np.sqrt(Kdiag*Kdiag.T)
    K_norm = old_div(K,normalizers)
    return K_norm










