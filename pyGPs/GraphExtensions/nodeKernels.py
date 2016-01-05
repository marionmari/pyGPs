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


import numpy as np
import scipy.linalg as spla
import scipy.sparse as spsp


def normLap(A):
    '''
    Normalized Laplacian

    :param A: adjacency matrix
    :return: kernel matrix
    '''
    I = np.identity(A.shape[0])
    d = A.sum(axis=0)
    d = np.sqrt(old_div(1.,d))    
    D = np.diag(d)
    L = I - np.dot( np.dot(D,A),D )
    return L


def regLapKernel(A, sigma=1):
    '''
    Regularized Laplacian Kernel 
        
    :param A: adjacency matrix
    :param sigma: hyperparameter sigma
    :return: kernel matrix
    '''
    I = np.identity(A.shape[0])
    L = normLap(A)
    K = np.linalg.inv( I+(sigma**2)*L )
    return K


def psInvLapKernel(A):
    '''
    Pseudo inverse of the normalized Laplacian.

    :param A: adjacency matrix
    :return: kernel matrix
    '''
    L = normLap(A)
    K = np.linalg.pinv(L)
    return K


def diffKernel(A, beta=0.5):
    '''
    Diffusion Process Kernel 
        
    K = exp(beta * H), where H = -L = A-D
    
    K = Q exp(beta * Lambda) Q.T
        
    :param A: adjacency matrix
    :param beta: hyperparameter beta
    :return: kernel matrix
    '''
    A = np.array(A) # make sure that A is a numpy array!!
    H = A - np.diag(np.sum(A, axis=1))
    w, Q = np.linalg.eigh(H)
    Lambda = np.diag(np.exp(beta*w))
    K = np.dot(np.dot(Q, Lambda), Q.T)
    return K


def VNDKernel(A, alpha=0.5):
    '''
    Von Neumann Diffusion Kernel on graph (Zhou et al., 2004)
    (also label spreading kernel)
        
    K = (I - alpha*S)^-1, where S = D^-1/2*A*D^-1/2
        
    :param A: adjacency matrix
    :param alpha: hyperparameter alpha
    :return: kernel matrix
    ''' 
    I = np.identity(A.shape[0])
    d = A.sum(axis=0)
    d = np.sqrt(old_div(1.,d))    
    D = np.diag(d)
    S = np.dot( np.dot(D,A),D )
    K = np.linalg.inv( I - alpha*S )
    return K


def rwKernel(A, p=1, a=2):
    '''
    p-step Random Walk Kernel with a>1
       
    K = (aI-L)^p, p>1 and L is the normalized Laplacian 
     
    :param A: adjacency matrix
    :param p: step parameter
    :param a: hyperparameter a
    :return: kernel matrix
    '''
    if type(p) != int:
        p = int(p)
    if p < 1:
        raise Exception('Step parameter p needs to be larger than 0.')
    if a <= 1:
        a=1.0001
    I = np.identity(A.shape[0])
    L = normLap(A)
    K = np.linalg.matrix_power( a*I - L, p)
    return K



def cosKernel(A):
    '''
    Cosine Kernel (also Inverse Cosine Kernel)
    
    K = cos (L*pi/4), where L is the normalized Laplacian
                
    :param A: adjacency matrix
    :return: kernel matrix
    '''
    L = normLap(A)
    K = np.cos(L*np.pi/4)   
    return K 



