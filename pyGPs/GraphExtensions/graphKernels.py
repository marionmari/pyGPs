from __future__ import division
from __future__ import print_function
from builtins import str
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


# Created on April, 2014
# @author: marion, shan

import scipy.sparse as spsp
import numpy as np
import matplotlib.pyplot as plt
    
def propagationKernel(A, l, gr_id, h_max, w, p, ktype=None, SUM=True, VIS=False, showEachStep=False):
    '''
    Propagation kernel for graphs as described in: 
    Neumann, M., Patricia, N., Garnett, R., Kersting, K.: Efficient Graph Kernels by 
    Randomization. In: P.A. Flach, T.D. Bie, N. Cristianini (eds.) ECML/PKDD, Notes in 
    Computer Science, vol. 7523, pp. 378-393. Springer (2012).

    :param A: adjacency matrix (num_nodes x num_nodes)
    :param l: label array (num_nodes x 1); values [1,...,k] or -1 for unlabeled nodes 
              OR label array (num_nodes x num_labels); values [0,1], unlabeled nodes have only 0 entries
    :param gr_id: graph indicator array (num_nodes x 1); values [0,..,n]
    :param h_max: number of iterations
    :param w: bin widths parameter
    :param p: distance ('tv', 'hellinger', 'L1', 'L2')
    :param ktype: type of propagation kernel ['diffusion', 'label_propagation', 'label_spreading', 'belief_propagation']
    
    :return: kernel matrix 
    '''
    # ToDO: SPLIT Varible label array and label probability
        
    # CAUTION: number of labels (k) > 1 !!! 
    
    #===========================================================================
    # ## Propagation Kernel COMPUTATION
    #===========================================================================
    num_graphs = gr_id.max().astype(int)
    num_nodes = A.shape[0]
    if l.shape[1]==1:
        num_labels = l.max().astype(int)
    else:
        num_labels = l.shape[1]
 
    #===========================================================================
    # ## INITIALIZATION
    #===========================================================================
    
    ## CHECK and ADJUST sparseness
    if not spsp.issparse(A): 
        A = spsp.csr_matrix(A)
    else:
        if not spsp.isspmatrix_csr(A):     
            A = A.tocsr()
    # MAKE sure A is float!        
    A = A.astype(float)                  
    ## INITIALIZE kernel matrix
    K = np.ndarray(shape=(num_graphs,num_graphs,h_max+1), dtype=float)
    ## INITIALIZE label probabilities of labeled nodes (num_nodes x num_labels)
    if l.shape[1]==1:
        lab_prob = np.zeros((num_nodes,num_labels),dtype=np.float64)            
        for i in range(num_nodes):
            if l[i,0] > 0:
                lab_prob[i,l[i,0]-1] = 1
    else: 
        lab_prob = l
        if spsp.issparse(lab_prob):
            lab_prob = lab_prob.todense()
    lab_orig  = lab_prob.copy()      # lab_orig is the orignal label information       
    ## GET indices of labeled and unlabeled nodes
    idx_unlab = np.where(np.max(lab_prob, axis=1)!=1)[0]
    idx_all   = np.array(np.arange(0,A.shape[0]), ndmin=2)
    idx       = np.array(np.setdiff1d(idx_all,idx_unlab), ndmin=2)     
    ## INITIALIZE unlabeled/uninitialized nodes UNIFORMLY
    idx_unif = np.where(np.sum(lab_prob, axis=1)==0)[0]         
    if idx_unif.shape[0] != 0:
        lab_prob[idx_unif,:] = old_div(1.,lab_prob.shape[1]) 
    ## row normalize A -> transition matrix T
    if (ktype=='label_propagation') or (ktype=='label_diffusion'):
        row_sums = np.array(A.sum(axis=1))[:,0] 
        T = A.copy()
        T.data /= row_sums[A.nonzero()[0]]    # so A and T are sparse matrix

    #===========================================================================
    # ## PROPAGATION KERNEL ITERATIONS
    #===========================================================================
    for h in range(h_max+1):  
        print('ITERATION: ', h)
        if h > 0:
            ## LABEL UPDATE 
            if showEachStep:
                print('...computing LABEL UPDATE')
            
            if ktype == 'label_propagation':
                lab_prob[idx,:] = lab_orig[idx,:]   # PUSH BACK original LABELS
                lab_prob = T*lab_prob
                
            elif ktype == 'label_diffusion': 
                lab_prob = T*lab_prob
            
            elif ktype == 'label_spreading':
                # S = D^(-1/2)*W*D^(-1/2)
                # y(t+1) = alpha*S*y(t)+(1-alpha)*y(0)
                alpha = 0.8
                # compute S
                diag = np.array(A.sum(axis=1)).T**(old_div(-1,2))
                D = spsp.lil_matrix((A.shape[0], A.shape[1]), dtype=float) #
                D.setdiag(diag[0,:])
                D = D.tocsr()
                S = D*A*D   
                lab_prob = np.dot((alpha*S.todense()),lab_prob) + (1-alpha)*lab_orig
        
        ## COMPUTE hashvalues 
        if showEachStep:
            print('...computing hashvalues')
        # determine path to take depending on chosen distance
        use_cauchy = (p =='L1') or (p =='tv')
        take_sqrt  = (p =='hellinger') 
        if take_sqrt:
            lab_prob = np.sqrt(lab_prob)      
        # generate random projection vector
        v = np.random.normal(size=(lab_prob.shape[1], 1))
        if use_cauchy:
            # if X, Y are N(0, 1), then X / Y has a standard Cauchy distribution
            v /= np.random.normal(size=(lab_prob.shape[1], 1))           
        # random offset
        b = w * np.random.rand()         
        # compute hashes
        # hashLabels is a Vector with length: number of nodes (hashvalue for respective node)
        hashLabels = old_div((np.dot(lab_prob,v) + b), w)
        hashLabels = np.floor(hashLabels)                           # take floor
        uniqueHash, hashLabels = np.unique(hashLabels, return_inverse=True)  # map to consecutive integer from 0        
        ## COMPUTE kernel contribution 
        if showEachStep:
            print('...computing KERNEL contribution')                     
        # aggregate counts on graphs
        # counts is a matrix: number of graphs x number of hashlabels
        num_bins = len(uniqueHash)
        counts = np.zeros((num_graphs, num_bins))      # init counts matrix 
        # accumulate counts of hash labels                              
        for i in range(num_nodes):
            counts[ (gr_id[i,0]-1), hashLabels[i] ] +=1
            
        # compute base kernel (here: LINEAR kernel)
        K_h = np.dot(counts,counts.T)      
        
        ## SUM kernel contributions
        if SUM:
            if h == 0:
                K[:,:,h] = K_h 
            else:
                K[:,:,h] = K[:,:,h-1] + K_h 
        else:
            K[:,:,h] = K_h
        
        if showEachStep:
            print(K[:,:,h]) 

    ## VISUALIZE KERNELS
    if VIS:  
        for h in range(h_max+1): 
            K_h = K[:,:,h]
            plt.figure()
            plt.title('height:'+str(h))
            imgplot = plt.imshow(K_h)
            imgplot.set_interpolation('nearest')
            plt.colorbar()
        plt.show()
    
    return K
    
 
    
