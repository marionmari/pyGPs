from __future__ import print_function
from builtins import str
from builtins import range

#! /usr/bin/env python
#coding=utf-8
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
from scipy.sparse.csc import csc_matrix
import pyGPs
from pyGPs.Validation import valid
from pyGPs.GraphExtensions import graphUtil,graphKernels

location = 'graphData/'
data = np.load(location+'MUTAG.npz')

# n = num of nodes
# N = num of graphs
# p = num of labels
A = csc_matrix( (data['adj_data'], data['adj_indice'], \
    data['adj_indptr']), shape=data['adj_shape'])     # n x n adjancy array (sparse matrix)  
gr_id = data['graph_ind']                             # n x 1 graph id array
node_label = data['responses']                        # n x 1 node label array
graph_label = data['labels']                          # N x 1 graph label array
N = graph_label.shape[0]                              # number of graphs)

graph_label = np.int8(graph_label)
for i in range(N):
    if graph_label[i,0] == 0:
        graph_label[i,0] -= 1

#===========================================================================
# COMPUTE PROPAGATION KERNELS
#===========================================================================
num_Iteration = 10
w = 1e-4
dist = 'tv'         # possible values: 'tv', 'hellinger'
np.random.seed(1)    # set random seed to get reproducible kernel matrices (to account for randomness in kernel average resutls over several returns of the experiment)    
K = graphKernels.propagationKernel(A, node_label, gr_id, num_Iteration, w, dist, 'label_diffusion', SUM=True, VIS=False, showEachStep=False) 

#----------------------------------------------------------------------
# Cross Validation
#----------------------------------------------------------------------
print('...GP prediction (10-fold CV)')

for t in range(num_Iteration+1):
    ACC = []           # accuracy
    
    print('number of kernel iterations =', t)
    Matrix = K[:,:,t]
    # normalize kernel matrix (not useful for MUTAG)
    # Matrix = graphUtil.normalizeKernel(Matrix)
            
    # start cross-validation for this t
    for index_train, index_test in valid.k_fold_index(N, K=10):
        
        y_train = graph_label[index_train,:]
        y_test  = graph_label[index_test,:]

        n1 = len(index_train)
        n2 = len(index_test)        
        
        model = pyGPs.GPC()
        M1,M2 = graphUtil.formKernelMatrix(Matrix, index_train, index_test)
        k = pyGPs.cov.Pre(M1,M2)
        model.setPrior(kernel=k)

        # gp
        x_train = np.zeros((n1,1)) 
        x_test = np.zeros((n2,1))       
        model.getPosterior(x_train, y_train)
        model.predict(x_test)
        predictive_class = np.sign(model.ym)

        # evaluation 
        acc = valid.ACC(predictive_class, y_test)   
        ACC.append(acc)
    
    
    print('Accuracy: ', np.round(np.mean(ACC),2), '('+str(np.round(np.std(ACC),2))+')')
     
  
