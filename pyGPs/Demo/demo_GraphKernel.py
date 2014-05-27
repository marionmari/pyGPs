#! /usr/bin/env python
#coding=utf-8

import numpy as np
from scipy.io import loadmat
from pyGPs.Core import *
from pyGPs.Valid import valid
from pyGPs.GraphStuff.graph_util import *
from pyGPs.GraphStuff.graph_kernels import *

allSets = ['MUTAG.mat', 'DD.mat', 'ENZYMES.mat', 'NCI1.mat', 'NCI109.mat']
location = 'data_for_demo/graphData/'
data = loadmat(location+allSets[2])

# n = num of nodes
# N = num of graphs
# p = num of labels
A = data['A']                    # n x n adjancy array (sparse matrix)  
gr_id = data['graph_ind']        # n x 1 graph id array
node_label = data['responses']   # n x 1 node label array
graph_label = data['labels']     # N x 1 graph label array
N = graph_label.shape[0]         # number of graphs)

print graph_label
graph_label = np.int8(graph_label)
for i in xrange(N):
    if graph_label[i,0] == 0:
        graph_label[i,0] -= 1

#===========================================================================
# COMPUTE PROPAGATION KERNELS
#===========================================================================
max_height = np.arange(1,11)
K = propagationKernel(A, node_label, gr_id, 10, 'label_propagation', SUM=True, VIS=False, showEachStep=True) 


#----------------------------------------------------------------------
# Cross Validation
#----------------------------------------------------------------------
ACC = []           # accuracy 

for T in max_height:
    print 'max height(T) is', T
    Matrix = K[:,:,T]
            
    # start cross-validation for this T
    for index_train, index_test in valid.k_fold_index(N, K=10):
        
        y_train = graph_label[index_train,:]
        y_test  = graph_label[index_test,:]
        n1 = len(index_train)
        n2 = len(index_test)        
        
        model = gp.GPC()
        M1,M2 = form_kernel_matrix(Matrix, index_train, index_test)
        k = cov.Pre(M1,M2)
        model.setPrior(kernel=k)
        
        # gp
        x_train = np.zeros((n1,1)) 
        x_test = np.zeros((n2,1))       
        model.train(np.zeros((n1,1)), y_train)
        model.predict(x_test)
        
        # evaluation 
        predictive_class = np.sign(model.ym)
        acc = valid.ACC(predictive_class, y_test)   
        ACC.append(acc) 
    
    print '\nAccuracy: ', np.round(np.mean(ACC),2), '('+str(np.round(np.std(ACC),2))+')'
        
