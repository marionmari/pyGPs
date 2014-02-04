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
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 30/09/2013
#================================================================================

import scipy.spatial as spspatial
import scipy.sparse as spsp
import numpy as np

def form_knn_graph(pc,k):
    ''' INPUT:     pc    n by D data matrix
                         where n is num_points
                               D is num_dimensions
            
        OUTPUT:    A     adjacency matrix '''

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


def form_kernel_matrix(M, indice_train, indice_test):
    ''' format precomputed kernel matrix into two matrix,
        which fit the parameters of method in pyGP

        INPUT:    M             n by n precomputed kernel matrix
                  indice_train  list of indice of training examples
                  indice_test   list of indice of test examples
        
        OUTPUT:   M1            train+1 by test matrix
                                     where the last row is the diagonal of test-test covariance
                  M2            train by train matrix

            '''
    train_test = M[indice_train,:][:,indice_test]
    test_test = M[indice_test,:][:,indice_test]
    dia = np.diag(test_test)
    test_dia = np.reshape(dia, (1,dia.shape[0]))
    M1 = np.concatenate((train_test,test_dia))
    M2 = M[indice_train,:][:,indice_train]
    return M1,M2










