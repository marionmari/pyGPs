from __future__ import print_function
import numpy as np
from scipy.io import loadmat
from scipy.sparse.csc import csc_matrix

data = loadmat('MUTAG.mat')

# n = num of nodes
# N = num of graphs
# p = num of labels
A = data['A']                    # n x n adjancy array (sparse matrix)  
gr_id = data['graph_ind']        # n x 1 graph id array
node_label = data['responses']   # n x 1 node label array
graph_label = data['labels']     # N x 1 graph label array

print(A.shape)
print(A.indices)
print(A.data)
print(A.indptr)
print(type(A))


np.savez("MUTAG", graph_ind=gr_id, responses=node_label, labels=graph_label, adj_data=A.data,\
         adj_indice=A.indices, adj_indptr=A.indptr, adj_shape=A.shape)




