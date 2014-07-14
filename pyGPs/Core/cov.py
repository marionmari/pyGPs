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
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 18/02/2014
#================================================================================
# covariance functions to be use by Gaussian process functions. There are two
# different kinds of covariance functions: simple and composite:
#
# simple cov functions:
#
# Poly          - Polynomial covariance function
# PiecePoly     - Piecewise polynomial kernel with compact support.
# RBF           - Squared Exponential kernel.
# RBFunit       - Squared Exponential kernel with unit magnitude.
# RBFard        - Squared Exponential kernel with Automatic Relevance Determination.
# Const         - Constant kernel.
# Linear        - Linear kernel.
# LINard        - Linear covariance function with Automatic Relevance Detemination.
# Matern        - Matern covariance function.
# Periodic      - Stationary kernel for a smooth periodic function.
# Noise         - Independent covariance function, i.e "white noise".
# RQ            - Rational Quadratic covariance function with isotropic distance measure.
# RQard         - Rational Quadratic covariance function with ARD distance measure.
# Pre           - Precomputed kernel matrix.
# 
# composite covariance functions:
#
#   ScaleOfKernel     - scaled version of a covariance function
#   ProductOfKernel   - products of covariance functions
#   SumOfKernel       - sums of covariance functions
#   FITCOfKernel      - Covariance function to be used together with the FITC approximation
#
#
# This is a object-oriented python implementation of gpml functionality 
# (Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18).
# based on the functional-version of python implementation
# (Copyright (c) by Marion Neumann and Daniel Marthaler, 20/05/2013)
# 
# Copyright (c) by Marion Neumann and Shan Huang, 30/09/2013


import numpy as np
import math
import scipy.spatial.distance as spdist


class Kernel(object):
    """
    This is a base class of Kernel functions
    there is no computation in this class, it just defines rules about a kernel class should have
    each covariance function will inherit it and implement its own behaviour
    """
    def __init__(self):
        self.hyp = []
        self.para = []

    def getCovMatrix(self,x=None,z=None,mode=None):
        '''
        Return the specific covariance matrix according to input mode

        :param x: training data
        :param z: test data
        :param str mode: 'self_test' return self covariance matrix of test data(test by 1). 
                         'train' return training covariance matrix(train by train).
                         'cross' return cross covariance matrix between x and z(train by test)

        :return: the corresponding covariance matrix
        '''
        pass

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        '''
        Compute derivatives wrt. hyperparameters according to input mode

        :param x: training data
        :param z: test data
        :param str mode: 'self_test' return self derivative matrix of test data(test by 1). 
                         'train' return training derivative matrix(train by train).
                         'cross' return cross derivative matrix between x and z(train by test)
        :param int der: index of hyperparameter whose derivative to be computed

        :return: the corresponding derivative matrix
        '''
        pass

    # overloading 
    def __add__(self,cov):
        '''
        Overloading + operator.
        
        :param cov: covariance function
        :return: an instance of SumOfKernel
        '''
        return SumOfKernel(self,cov)

    # overloading 
    def __mul__(self,other):
        '''
        Overloading * operator.
        Using * for both multiplication with scalar and product of kernels
        depending on the type of the two objects.
        
        :param other: covariance function as product or int/float as scalar
        :return: an instance of ScaleOfKernel or ProductOfKernel
        '''
        if isinstance(other, int) or isinstance(other, float):
            return ScaleOfKernel(self,other)
        elif isinstance(other, Kernel):
            return ProductOfKernel(self,other)
        else:
            print "only numbers and Kernels are supported operand types for *"

    # overloading 
    __rmul__ = __mul__

    def fitc(self,inducingInput):
        '''
        Covariance function to be used together with the FITC approximation.
        Setting FITC gp model will implicitly call this method.

        :return: an instance of FITCOfKernel
        '''
        return FITCOfKernel(self,inducingInput)
    
    # can be replaced by spdist from scipy
    def sq_dist(self, a, b=None):
        '''Compute a matrix of all pairwise squared distances
        between two sets of vectors, stored in the row of the two matrices:
        a (of size n by D) and b (of size m by D).'''
        n = a.shape[0]
        D = a.shape[1]
        m = n    
        if b == None:
            b = a.transpose()
        else:
            m = b.shape[0]
            b = b.transpose()
        C = np.zeros((n,m))
        for d in range(0,D):
            tt  = a[:,d]
            tt  = tt.reshape(n,1)
            tem = np.kron(np.ones((1,m)), tt)
            tem = tem - np.kron(np.ones((n,1)), b[d,:])
            C   = C + tem * tem 
        return C

              
        
class ProductOfKernel(Kernel):
    '''Product of two kernel function.'''
    def __init__(self,cov1,cov2):
        self.cov1 = cov1
        self.cov2 = cov2
        self._hyp = cov1.hyp + cov2.hyp

    def sethyp(self,hyp):
        assert len(hyp) == len(self._hyp)
        len1 = len(self.cov1.hyp)
        self._hyp = hyp 
        self.cov1.hyp = self._hyp[:len1]
        self.cov2.hyp = self._hyp[len1:]
    def gethyp(self):
        return self._hyp
    hyp = property(gethyp,sethyp)

    def getCovMatrix(self,x=None,z=None,mode=None):
        A = self.cov1.getCovMatrix(x,z,mode) * self.cov2.getCovMatrix(x,z,mode)
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        if der < len(self.cov1.hyp):
            A = self.cov1.getDerMatrix(x,z,mode,der) * self.cov2.getCovMatrix(x,z,mode)
        elif der < len(self.hyp):
            der2 = der - len(self.cov1.hyp)
            A = self.cov2.getDerMatrix(x,z,mode,der2) * self.cov1.getCovMatrix(x,z,mode)
        else:
            raise Exception("Error: der out of range for covProduct") 
        return A



class SumOfKernel(Kernel):
    '''Sum of two kernel function.'''
    def __init__(self,cov1,cov2):
        self.cov1 = cov1
        self.cov2 = cov2
        self._hyp = cov1.hyp + cov2.hyp
    def sethyp(self,hyp):
        assert len(hyp) == len(self._hyp)
        len1 = len(self.cov1.hyp)
        self._hyp = hyp 
        self.cov1.hyp = self._hyp[:len1]
        self.cov2.hyp = self._hyp[len1:]
    def gethyp(self):
        return self._hyp
    hyp = property(gethyp,sethyp)

    def getCovMatrix(self,x=None,z=None,mode=None):
        A = self.cov1.getCovMatrix(x,z,mode) + self.cov2.getCovMatrix(x,z,mode)
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        if der < len(self.cov1.hyp):
            A = self.cov1.getDerMatrix(x,z,mode,der) 
        elif der < len(self.hyp):
            der2 = der - len(self.cov1.hyp)
            A = self.cov2.getDerMatrix(x,z,mode,der2) 
        else:
            raise Exception("Error: der out of range for covSum") 
        return A



class ScaleOfKernel(Kernel):
    '''Scale of a kernel function.'''
    def __init__(self,cov,scalar):
        self.cov = cov
        if cov.hyp:
            self._hyp = [scalar] + cov.hyp 
        else:
            self._hyp = [scalar]
    def sethyp(self,hyp):
        assert len(hyp) == len(self._hyp)
        self._hyp = hyp 
        self.cov.hyp = self._hyp[1:]
    def gethyp(self):
        return self._hyp
    hyp = property(gethyp,sethyp)

    def getCovMatrix(self,x=None,z=None,mode=None):
        sf2 = np.exp(self.hyp[0])                     # scale parameter  
        A = sf2 * self.cov.getCovMatrix(x,z,mode)     # accumulate cov 
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        sf2 = np.exp(self.hyp[0])                     # scale parameter  
        if der == 0:                                  # compute derivative w.r.t. sf2
            A = 2. * sf2 * self.cov.getCovMatrix(x,z,mode)
        else:                                 
            A = sf2 * self.cov.getDerMatrix(x,z,mode,der-1) 
        return A
     


class FITCOfKernel(Kernel):
    '''
    Covariance function to be used together with the FITC approximation.
    The function allows for more than one output argument and does not respect the
    interface of a proper covariance function. 
    Instead of outputing the full covariance, it returns cross-covariances between
    the inputs x, z and the inducing inputs xu as needed by infFITC
    '''
    def __init__(self,cov,inducingInput):
        self.inducingInput = inducingInput
        self.covfunc = cov
        self._hyp = cov.hyp

    def getHyp(self):
        return self._hyp
    def setHyp(self, hyp):
        self._hyp = hyp
        self.covfunc.hyp = hyp
    hyp = property(getHyp,setHyp)

    def getCovMatrix(self,x=None,z=None,mode=None):
        xu = self.inducingInput
        if x != None:
            try:
                assert(xu.shape[1] == x.shape[1])
            except AssertionError:
                raise Exception('Dimensionality of inducing inputs must match training inputs') 
        if mode == 'self_test':           # self covariances for the test cases
            K = self.covfunc.getCovMatrix(z=z,mode='self_test')
            return K
        elif mode == 'train':             # compute covariance matix for training set
            K   = self.covfunc.getCovMatrix(z=x,mode='self_test')
            Kuu = self.covfunc.getCovMatrix(x=xu,mode='train')
            Ku  = self.covfunc.getCovMatrix(x=xu,z=x,mode='cross')
            return K, Kuu, Ku
        elif mode == 'cross':             # compute covariance between data sets x and z 
            K = self.covfunc.getCovMatrix(x=xu,z=z,mode='cross')
            return K 

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        xu = self.inducingInput
        if x!= None:
            try:
                assert(xu.shape[1] == x.shape[1])
            except AssertionError:
                raise Exception('Dimensionality of inducing inputs must match training inputs') 
        if mode == 'self_test':           # self covariances for the test cases
            K = self.covfunc.getDerMatrix(z=z,mode='self_test',der=der)
            return K
        elif mode == 'train':             # compute covariance matix for training set
            K   = self.covfunc.getDerMatrix(z=x,mode='self_test',der=der)
            Kuu = self.covfunc.getDerMatrix(x=xu,mode='train',der=der)
            Ku  = self.covfunc.getDerMatrix(x=xu,z=x,mode='cross',der=der)
            return K, Kuu, Ku
        elif mode == 'cross':             # compute covariance between data sets x and z 
            K = self.covfunc.getDerMatrix(x=xu,z=z,mode='cross',der=der)
            return K 
        


class Poly(Kernel):
    '''
    Polynomial covariance function. hyp = [ log_c, log_sigma ]

    :param log_c: inhomogeneous offset. 
    :param log_sigma: signal deviation. 
    :param d: degree of polynomial (not treated as hyperparameter, i.e. will not be trained). 
    '''
    def __init__(self, log_c=0., d=2, log_sigma=0. ):
        self.hyp = [log_c, log_sigma]
        self.para = [d] 

    def getCovMatrix(self,x=None,z=None,mode=None):
        c   = np.exp(self.hyp[0])             # inhomogeneous offset
        sf2 = np.exp(2.*self.hyp[1])          # signal variance
        ord = self.para[0]                    # order of polynomial
        if np.abs(ord-np.round(ord)) < 1e-8:  # remove numerical error from format of parameter
            ord = int(round(ord))
        assert(ord >= 1.)                     # only nonzero integers for ord
        ord = int(ord) 
        if mode == 'self_test':               # self covariances for the test cases
            nn,D = z.shape
            A = np.reshape(np.sum(z*z,1), (nn,1))
        elif mode == 'train':                 # compute covariance matix for dataset x
            A = np.dot(x,x.T)
        elif mode == 'cross':                 # compute covariance between data sets x and z 
            A = np.dot(x,z.T) 
        A = sf2 * (c + A)**ord
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        c   = np.exp(self.hyp[0])             # inhomogeneous offset
        sf2 = np.exp(2.*self.hyp[1])          # signal variance
        ord = self.para[0]                    # order of polynomial
        if np.abs(ord-np.round(ord)) < 1e-8:  # remove numerical error from format of parameter
            ord = int(round(ord))
        assert(ord >= 1.)                     # only nonzero integers for ord
        ord = int(ord) 
        if mode == 'self_test':               # self covariances for the test cases
            nn,D = z.shape
            A = np.reshape(np.sum(z*z,1), (nn,1))
        elif mode == 'train':                 # compute covariance matix for dataset x
            A = np.dot(x,x.T)
        elif mode == 'cross':                 # compute covariance between data sets x and z 
            A = np.dot(x,z.T) 
        if der == 0:                          # compute derivative matrix wrt 1st parameter             
            A = c * ord * sf2 * (c+A)**(ord-1)
        elif der == 1:                        # compute derivative matrix wrt 2nd parameter
            A = 2. * sf2 * (c + A)**ord
        elif der == 2:                        # no derivative wrt 3rd parameter
            A = np.zeros_like(A)              # do nothing (d is not learned)
        else:
            raise Exception("Wrong derivative entry in Poly")
        return A
        


class PiecePoly(Kernel):
    '''
    Piecewise polynomial kernel with compact support. 
    hyp = [log_ell, log_sigma]

    :param log_ell: characteristic length scale. 
    :param log_sigma: signal deviation. 
    :param v: degree v will be rounded to 0,1,2,or 3. (not treated as hyperparameter, i.e. will not be trained). 
    '''
    def __init__(self, log_ell=0., v=2, log_sigma=0. ):
        self.hyp = [log_ell, log_sigma]
        self.para = [v]

    def ppmax(self,A,B):
        return np.maximum(A,B*np.ones_like(A))

    def func(self,v,r,j):
        if v == 0:
            return 1
        elif v == 1:
            return ( 1. + (j+1) * r )
        elif v == 2:
            return ( 1. + (j+2)*r + (j*j + 4.*j+ 3)/3.*r*r )
        elif v == 3:
            return ( 1. + (j+3)*r + (6.*j*j+36.*j+45.)/15.*r*r + (j*j*j+9.*j*j+23.*j+15.)/15.*r*r*r )
        else:
             raise Exception (["Wrong degree in PiecePoly.  Should be 0,1,2 or 3, is " + str(v)])

    def dfunc(self,v,r,j):
        if v == 0:
            return 0
        elif v == 1:
            return ( j+1 )
        elif v == 2:
            return ( (j+2) + 2.*(j*j+ 4.*j+ 3.)/3.*r )
        elif v == 3:
            return ( (j+3) + 2.*(6.*j*j+36.*j+45.)/15.*r + (j*j*j+9.*j*j+23.*j+15.)/5.*r*r )
        else:
            raise Exception (["Wrong degree in PiecePoly.  Should be 0,1,2 or 3, is " + str(v)])

    def pp(self,r,j,v,func):
        return func(v,r,j)*(self.ppmax(1-r,0)**(j+v))

    def dpp(self,r,j,v,func,dfunc):
        return self.ppmax(1-r,0)**(j+v-1) * r * ( (j+v)*self.func(v,r,j) - self.ppmax(1-r,0) * self.dfunc(v,r,j) )

    def getCovMatrix(self,x=None,z=None,mode=None):
        if x != None:
            n, D = x.shape
        if z != None:
            nn, D = z.shape
        ell = np.exp(self.hyp[0])            # characteristic length scale
        sf2 = np.exp(2.*self.hyp[1])         # signal variance
        v   = self.para[0]                   # degree (v = 0,1,2 or 3 only)
        if np.abs(v-np.round(v)) < 1e-8:     # remove numerical error from format of parameter
            v = int(round(v))
        assert(int(v) in range(4))           # Only allowed degrees: 0,1,2 or 3
        v = int(v)
        j = np.floor(0.5*D) + v + 1
        if mode == 'self_test':              # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':                # compute covariance matix for dataset x
            A = np.sqrt( spdist.cdist(x/ell, x/ell, 'sqeuclidean') )
        elif mode == 'cross':                # compute covariance between data sets x and z 
            A = np.sqrt( spdist.cdist(x/ell, z/ell, 'sqeuclidean') )
        A = sf2 * self.pp(A,j,v,self.func)
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        if x != None:
            n, D = x.shape
        if z != None:
            nn, D = z.shape
        ell = np.exp(self.hyp[0])            # characteristic length scale
        sf2 = np.exp(2.*self.hyp[1])         # signal variance
        v   = self.para[0]                   # degree (v = 0,1,2 or 3 only)
        if np.abs(v-np.round(v)) < 1e-8:     # remove numerical error from format of parameter
            v = int(round(v))
        assert(int(v) in range(4))           # Only allowed degrees: 0,1,2 or 3
        v = int(v)
        j = np.floor(0.5*D) + v + 1
        if mode == 'self_test':              # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':                # compute covariance matix for dataset x
            A = np.sqrt( spdist.cdist(x/ell, x/ell, 'sqeuclidean') )
        elif mode == 'cross':                # compute covariance between data sets x and z 
            A = np.sqrt( spdist.cdist(x/ell, z/ell, 'sqeuclidean') )
        if der == 0:                            # compute derivative matrix wrt 1st parameter
            A = sf2 * self.dpp(A,j,v,self.func,self.dfunc)
        elif der == 1:                          # compute derivative matrix wrt 2nd parameter
            A = 2. * sf2 * self.pp(A,j,v,self.func)
        elif der == 2:                          # wants to compute derivative wrt order
            A = np.zeros_like(A)
        else:
            raise Exception("Wrong derivative entry in PiecePoly")
        return A



class RBF(Kernel):
    '''
    Squared Exponential kernel with isotropic distance measure. hyp = [log_ell, log_sigma]

    :param log_ell: characteristic length scale. 
    :param log_sigma: signal deviation. 
    '''
    def __init__(self, log_ell=0., log_sigma=0.):
        self.hyp = [log_ell, log_sigma]

    def getCovMatrix(self,x=None,z=None,mode=None):
        ell = np.exp(self.hyp[0])         # characteristic length scale
        sf2 = np.exp(2.*self.hyp[1])      # signal variance
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for training set
            A = spdist.cdist(x/ell,x/ell,'sqeuclidean')
        elif mode == 'cross':             # compute covariance between data sets x and z 
            A = spdist.cdist(x/ell,z/ell,'sqeuclidean') 
        A = sf2 * np.exp(-0.5*A)
        return A


    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        ell = np.exp(self.hyp[0])         # characteristic length scale
        sf2 = np.exp(2.*self.hyp[1])      # signal variance
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            A = spdist.cdist(x/ell,x/ell,'sqeuclidean')
        elif mode == 'cross':             # compute covariance between data sets x and z 
            A = spdist.cdist(x/ell,z/ell,'sqeuclidean') 
        if der == 0:    # compute derivative matrix wrt 1st parameter
            A = sf2 * np.exp(-0.5*A) * A
        elif der == 1:  # compute derivative matrix wrt 2nd parameter
            A = 2. * sf2 * np.exp(-0.5*A)
        else:
            raise Exception("Calling for a derivative in RBF that does not exist")
        return A
        


class RBFunit(Kernel):
    '''
    Squared Exponential kernel with isotropic distance measure with unit magnitude.
    i.e signal variance is always 1. hyp = [ log_ell ]

    :param log_ell: characteristic length scale. 
    '''
    def __init__(self, log_ell=0.):
        self.hyp = [log_ell]

    def getCovMatrix(self,x=None,z=None,mode=None):
        ell = np.exp(self.hyp[0])         # characteristic length scale
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            A = spdist.cdist(x/ell,x/ell,'sqeuclidean')
        elif mode == 'cross':             # compute covariance between data sets x and z 
            A = spdist.cdist(x/ell,z/ell,'sqeuclidean') 
        A = np.exp(-0.5*A)
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        ell = np.exp(self.hyp[0])         # characteristic length scale
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            A = spdist.cdist(x/ell,x/ell,'sqeuclidean')
        elif mode == 'cross':             # compute covariance between data sets x and z 
            A = spdist.cdist(x/ell,z/ell,'sqeuclidean') 
        if der == 0:           # compute derivative matrix wrt 1st parameter
            A = np.exp(-0.5*A) * A
        else:
            raise Exception("Wrong derivative index in RDFunit")
        return A



class RBFard(Kernel):
    '''
    Squared Exponential kernel with Automatic Relevance Determination.
    hyp = log_ell_list + [log_sigma]

    :param D: dimension of pattern. set if you want default ell, which is 1 for each dimension.
    :param log_ell_list: characteristic length scale for each dimension.
    :param log_sigma: signal deviation. 
    '''
    def __init__(self, D=None, log_ell_list=None, log_sigma=0.):
        if log_ell_list == None:
            self.hyp = [0. for i in xrange(D)] + [log_sigma]
        else:
            self.hyp = log_ell_list + [log_sigma]

    def getCovMatrix(self,x=None,z=None,mode=None):
        if x != None:
            n, D = x.shape
        if z != None:
            nn, D = z.shape
        ell = 1./np.exp(self.hyp[0:D])    # characteristic length scale
        sf2 = np.exp(2.*self.hyp[D])      # signal variance
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            tem = np.dot(np.diag(ell),x.T).T
            A = spdist.cdist(tem,tem,'sqeuclidean')
        elif mode == 'cross':             # compute covariance between data sets x and z 
            A = spdist.cdist(np.dot(np.diag(ell),x.T).T,np.dot(np.diag(ell),z.T).T,'sqeuclidean')
        A = sf2*np.exp(-0.5*A)
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        if x != None:
            n, D = x.shape
        if z != None:
            nn, D = z.shape
        ell = 1./np.exp(self.hyp[0:D])    # characteristic length scale
        sf2 = np.exp(2.*self.hyp[D])      # signal variance
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            tem = np.dot(np.diag(ell),x.T).T
            A = spdist.cdist(tem,tem,'sqeuclidean')
        elif mode == 'cross':             # compute covariance between data sets x and z 
            A = spdist.cdist(np.dot(np.diag(ell),x.T).T,np.dot(np.diag(ell),z.T).T,'sqeuclidean')
        A = sf2*np.exp(-0.5*A)
        if der < D:                       # compute derivative matrix wrt length scale parameters
            if mode == 'self_test':
                A = A*0
            elif mode == 'train':
                tem = np.atleast_2d(x[:,der])/ell[der]
                A *= spdist.cdist(tem,tem,'sqeuclidean')
            elif mode == 'cross':
                A *= spdist.cdist(np.atleast_2d(x[:,der]).T/ell[der],np.atleast_2d(z[:,der]).T/ell[der],'sqeuclidean')
        elif der == D:                    # compute derivative matrix wrt magnitude parameter
            A = 2.*A
        else:
            raise Exception("Wrong derivative index in RDFard") 
        return A



class Const(Kernel):
    '''
    Constant kernel. hyp = [ log_sigma ]

    :param log_sigma: signal deviation. 
    '''
    def __init__(self, log_sigma=0.):
        self.hyp = [log_sigma]

    def getCovMatrix(self,x=None,z=None,mode=None):
        sf2 = np.exp(self.hyp[0])         # s2
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = sf2 * np.ones((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            n,D = x.shape
            A = sf2 * np.ones((n,n))
        elif mode == 'cross':             # compute covariance between data sets x and z 
            n,D  = x.shape
            nn,D = z.shape
            A = sf2 * np.ones((n,nn))
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        sf2 = np.exp(self.hyp[0])         # s2
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = sf2 * np.ones((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            n,D = x.shape
            A = sf2 * np.ones((n,n))
        elif mode == 'cross':             # compute covariance between data sets x and z 
            n,D  = x.shape
            nn,D = z.shape
            A = sf2 * np.ones((n,nn))
        if der == 0:                      # compute derivative matrix wrt sf2
            A = 2. * A
        else:
            raise Exception("Wrong derivative entry in covConst")
        return A
        


class Linear(Kernel):
    '''
    Linear kernel. hyp = [ log_sigma ].
    '''
    def __init__(self, log_sigma=0.):
        self.hyp = [ log_sigma ]

    def getCovMatrix(self,x=None,z=None,mode=None):
        sf2 = np.exp(self.hyp[0])         # s2
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.reshape(np.sum(z*z,1), (nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            n,D = x.shape
            A = np.dot(x,x.T) + np.eye(n)*1e-16    # required for numerical accuracy
        elif mode == 'cross':             # compute covariance between data sets x and z 
            A = np.dot(x,z.T)
        A = sf2 * A
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        sf2 = np.exp(self.hyp[0])         # s2
        if der == 0:
            if mode == 'self_test':           # self covariances for the test cases
                nn,D = z.shape
                A = np.reshape(np.sum(z*z,1), (nn,1))
            elif mode == 'train':             # compute covariance matix for dataset x
                n,D = x.shape
                A = np.dot(x,x.T) + np.eye(n)*1e-16    # required for numerical accuracy
            elif mode == 'cross':             # compute covariance between data sets x and z 
                A = np.dot(x,z.T)
            A = 2 * sf2 * A
        else:
            raise Exception("Wrong derivative index in covLinear")
        return A



class LINard(Kernel):
    '''
    Linear covariance function with Automatic Relevance Detemination.
    hyp = log_ell_list 

    :param D: dimension of training data. Set if you want default ell, which is 1 for each dimension.
    :param log_ell_list: characteristic length scale for each dimension.
    '''
    def __init__(self, D=None, log_ell_list=None):
        if log_ell_list == None:
            self.hyp = [0. for i in xrange(D)]
        else:
            self.hyp = log_ell_list

    def getCovMatrix(self,x=None,z=None,mode=None):
        ell = np.exp(self.hyp)            # ARD parameters
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.reshape(np.sum(z*z,1), (nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            A = np.dot(x,x.T) 
        elif mode == 'cross':             # compute covariance between data sets x and z
            z = np.dot(z,np.diag(1./ell))
            A = np.dot(x,z.T) 
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        ell = np.exp(self.hyp)            # ARD parameters
        if x != None:
            n, D = x.shape
        if z != None:
            nn, D = z.shape
        if der < D:
            if mode == 'self_test':
                tem = np.atleast_2d(z[:,der]).T
                A = -2.* tem * tem
            elif mode == 'train':
                A = -2.*np.dot(np.atleast_2d(x[:,der]).T,np.atleast_2d(x[:,der]))
            elif mode == 'cross':
                z = np.dot(z,np.diag(1./ell))
                A = -2.*np.dot(np.atleast_2d(x[:,der]).T, np.atleast_2d(z[:,der])) # cross covariances
        else:
            raise Exception("Wrong derivative index in covLINard")
        return A



class Matern(Kernel):
    '''
    Matern covariance function with nu = d/2 and isotropic distance measure. 
    For d=1 the function is also known as the exponential covariance function 
    or the Ornstein-Uhlenbeck covariance in 1d.
    d will be rounded to 1, 3, or 5.
    hyp = [ log_ell, log_sigma]
    
    :param d: d is 2 times nu. Can only be 1,3 or 5.
    :param log_ell: characteristic length scale. 
    :param log_sigma: signal deviation. 
    '''
    def __init__(self, log_ell=0., d=3, log_sigma=0. ):
        self.hyp = [ log_ell, log_sigma ]
        self.para = [d]

    def func(self,d,t):
        if d == 1:
            return 1
        elif d == 3:
            return 1 + t
        elif d == 5:
            return 1 + t*(1+t/3.)
        else:
            raise Exception("Wrong value for d in Matern")

    def dfunc(self,d,t):
        if d == 1:
            return 1
        elif d == 3:
            return t
        elif d == 5:
            return t*(1+t/3.)
        else:
            raise Exception("Wrong value for d in Matern")

    def mfunc(self,d,t):
        return self.func(d,t)*np.exp(-1.*t)

    def dmfunc(self,d,t):
        return self.dfunc(d,t)*t*np.exp(-1.*t)

    def getCovMatrix(self,x=None,z=None,mode=None):
        ell = np.exp(self.hyp[0])        # characteristic length scale
        sf2 = np.exp(2.* self.hyp[1])    # signal variance
        d   = self.para[0]               # 2 times nu
        if np.abs(d-np.round(d)) < 1e-8: # remove numerical error from format of parameter
            d = int(round(d))
        d = int(d)
        try:
            assert(d in [1,3,5])         # check for valid values of d
        except AssertionError:
            print "Warning: You specified d to be neither 1,3 nor 5. We set to d=3. "
            d = 3
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            x = np.sqrt(d)*x/ell   
            A = np.sqrt(spdist.cdist(x, x, 'sqeuclidean'))
        elif mode == 'cross':             # compute covariance between data sets x and z
            x = np.sqrt(d)*x/ell
            z = np.sqrt(d)*z/ell
            A = np.sqrt(spdist.cdist(x, z, 'sqeuclidean'))
        A = sf2 * self.mfunc(d,A)
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        ell = np.exp(self.hyp[0])        # characteristic length scale
        sf2 = np.exp(2.* self.hyp[1])    # signal variance
        d   = self.para[0]               # 2 times nu
        if np.abs(d-np.round(d)) < 1e-8: # remove numerical error from format of parameter
            d = int(round(d))
        d = int(d)
        try:
            assert(d in [1,3,5])         # check for valid values of d
        except AssertionError:
            print "Warning: You specified d to be neither 1,3 nor 5. We set to d=3. "
            d = 3
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            x = np.sqrt(d)*x/ell   
            A = np.sqrt(spdist.cdist(x, x, 'sqeuclidean'))
        elif mode == 'cross':             # compute covariance between data sets x and z
            x = np.sqrt(d)*x/ell
            z = np.sqrt(d)*z/ell
            A = np.sqrt(spdist.cdist(x, z, 'sqeuclidean'))
        A = sf2 * self.mfunc(d,A)
        if der == 0:                    # compute derivative matrix wrt 1st parameter
            A = sf2 * self.dmfunc(d,A)
        elif der == 1:                  # compute derivative matrix wrt 2nd parameter
            A = 2 * sf2 * self.mfunc(d,A)
        elif der == 2:                  # no derivative wrt 3rd parameter
            A = np.zeros_like(A)        # do nothing (d is not learned)
        else:
            raise Exception("Wrong derivative value in Matern")
        return A
        


class Periodic(Kernel):
    '''
    Stationary kernel for a smooth periodic function. 
    hyp = [ log_ell, log_p, log_sigma]

    :param log_p: period.
    :param log_ell: characteristic length scale. 
    :param log_sigma: signal deviation. 
    '''
    def __init__(self, log_ell=0., log_p=0., log_sigma=0. ): 
        self.hyp = [ log_ell, log_p, log_sigma]

    def getCovMatrix(self,x=None,z=None,mode=None):
        ell = np.exp(self.hyp[0])        # characteristic length scale
        p   = np.exp(self.hyp[1])        # period
        sf2 = np.exp(2.*self.hyp[2])     # signal variance
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            A = np.sqrt(spdist.cdist(x, x, 'sqeuclidean'))
        elif mode == 'cross':             # compute covariance between data sets x and z
            A = np.sqrt(spdist.cdist(x, z, 'sqeuclidean'))
        A = np.pi*A/p
        A = np.sin(A)/ell
        A = A * A
        A = sf2 *np.exp(-2.*A)
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        ell = np.exp(self.hyp[0])        # characteristic length scale
        p   = np.exp(self.hyp[1])        # period
        sf2 = np.exp(2.*self.hyp[2])     # signal variance
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            A = np.sqrt(spdist.cdist(x, x, 'sqeuclidean'))
        elif mode == 'cross':             # compute covariance between data sets x and z
            A = np.sqrt(spdist.cdist(x, z, 'sqeuclidean'))
        A = np.pi*A/p
        if der == 0:            # compute derivative matrix wrt 1st parameter
            A = np.sin(A)/ell
            A = A * A
            A = 4. *sf2 *np.exp(-2.*A) * A
        elif der == 1:          # compute derivative matrix wrt 2nd parameter
            R = np.sin(A)/ell
            A = 4 * sf2/ell * np.exp(-2.*R*R)*R*np.cos(A)*A
        elif der == 2:          # compute derivative matrix wrt 3rd parameter
            A = np.sin(A)/ell
            A = A * A
            A = 2. * sf2 * np.exp(-2.*A)
        else:
            raise Exception("Wrong derivative index in covPeriodic")
        return A
        


class Noise(Kernel):
    '''
    Independent covariance function, i.e "white noise", with specified variance.
    Normally NOT used anymore since noise is now added in liklihood.
    hyp = [ log_sigma ]

    :param log_sigma: signal deviation. 
    '''
    def __init__(self, log_sigma=0.):
        self.hyp = [log_sigma]

    def getCovMatrix(self,x=None,z=None,mode=None):
        tol = 1.e-9                       # Tolerance for declaring two vectors "equal"
        s2 = np.exp(2.*self.hyp[0])       # noise variance
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            n,D = x.shape
            A = np.eye(n)
        elif mode == 'cross':             # compute covariance between data sets x and z
            M = spdist.cdist(x, z, 'sqeuclidean')
            A = np.zeros_like(M,dtype=np.float)
            A[M < tol] = 1.
        A = s2*A
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        tol = 1.e-9                       # Tolerance for declaring two vectors "equal"
        s2 = np.exp(2.*self.hyp[0])       # noise variance
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            n,D = x.shape
            A = np.eye(n)
        elif mode == 'cross':             # compute covariance between data sets x and z
            M = spdist.cdist(x, z, 'sqeuclidean')
            A = np.zeros_like(M,dtype=np.float)
            A[M < tol] = 1.
        if der == 0:
            A = 2.*s2*A
        else:
            raise Exception("Wrong derivative index in covNoise")
        return A
        


class RQ(Kernel):
    ''' 
    Rational Quadratic covariance function with isotropic distance measure.
    hyp = [ log_ell, log_sigma, log_alpha ]
    
    :param log_ell: characteristic length scale. 
    :param log_sigma: signal deviation.
    :param log_alpha: shape parameter for the RQ covariance.
    '''
    def __init__(self, log_ell=0., log_sigma=0., log_alpha=0.):
        self.hyp = [ log_ell, log_sigma, log_alpha ]

    def getCovMatrix(self,x=None,z=None,mode=None):
        ell   = np.exp(self.hyp[0])       # characteristic length scale
        sf2   = np.exp(2.*self.hyp[1])    # signal variance
        alpha = np.exp(self.hyp[2])
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            D2 = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            D2 = spdist.cdist(x/ell, x/ell, 'sqeuclidean')
        elif mode == 'cross':             # compute covariance between data sets x and z
            D2 = spdist.cdist(x/ell, z/ell, 'sqeuclidean')
        A = sf2 * ( ( 1.0 + 0.5*D2/alpha )**(-alpha) )
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        ell   = np.exp(self.hyp[0])       # characteristic length scale
        sf2   = np.exp(2.*self.hyp[1])    # signal variance
        alpha = np.exp(self.hyp[2])
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            D2 = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            D2 = spdist.cdist(x/ell, x/ell, 'sqeuclidean')
        elif mode == 'cross':             # compute covariance between data sets x and z
            D2 = spdist.cdist(x/ell, z/ell, 'sqeuclidean')
        if der == 0:                # compute derivative matrix wrt 1st parameter
            A = sf2 * ( 1.0 + 0.5*D2/alpha )**(-alpha-1) * D2
        elif der == 1:              # compute derivative matrix wrt 2nd parameter
            A = 2.* sf2 * ( ( 1.0 + 0.5*D2/alpha )**(-alpha) )
        elif der == 2:              # compute derivative matrix wrt 3rd parameter
            K = ( 1.0 + 0.5*D2/alpha )
            A = sf2 * K**(-alpha) * (0.5*D2/K - alpha*np.log(K) )
        else:
            raise Exception("Wrong derivative index in covRQ")
        return A



class RQard(Kernel):
    '''
    Rational Quadratic covariance function with Automatic Relevance Detemination
    (ARD) distance measure.
    hyp = log_ell_list + [ log_sigma, log_alpha ]

    :param D: dimension of pattern. set if you want default ell, which is 0.5 for each dimension.
    :param log_ell_list: characteristic length scale for each dimension.
    :param log_sigma: signal deviation. 
    :param log_alpha: shape parameter for the RQ covariance.
    '''
    def __init__(self, D=None, log_ell_list=None, log_sigma=0., log_alpha=0.):
        if log_ell_list == None:
            self.hyp = [0. for i in xrange(D)] + [ log_sigma, log_alpha ]
        else:
            self.hyp = log_ell_list + [ log_sigma, log_alpha ]

    def getCovMatrix(self,x=None,z=None,mode=None):
        if x != None:
            n, D = x.shape
        if z != None:
            nn, D = z.shape
        ell = 1./np.exp(self.hyp[0:D])    # characteristic length scale
        sf2 = np.exp(2.*self.hyp[D])      # signal variance
        alpha = np.exp(self.hyp[D+1])
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            D2 = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            tmp = np.dot(np.diag(ell),x.T).T
            D2 = spdist.cdist(tmp, tmp, 'sqeuclidean')
        elif mode == 'cross':             # compute covariance between data sets x and z
            D2 = spdist.cdist(np.dot(np.diag(ell),x.T).T, np.dot(np.diag(ell),z.T).T, 'sqeuclidean')
        A = sf2 * ( ( 1.0 + 0.5*D2/alpha )**(-alpha) )
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        if x != None:
            n, D = x.shape
        if z != None:
            nn, D = z.shape
        ell = 1./np.exp(self.hyp[0:D])    # characteristic length scale
        sf2 = np.exp(2.*self.hyp[D])      # signal variance
        alpha = np.exp(self.hyp[D+1])
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            D2 = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            tmp = np.dot(np.diag(ell),x.T).T
            D2 = spdist.cdist(tmp, tmp, 'sqeuclidean')
        elif mode == 'cross':             # compute covariance between data sets x and z
            D2 = spdist.cdist(np.dot(np.diag(ell),x.T).T, np.dot(np.diag(ell),z.T).T, 'sqeuclidean')
        if der < D:
            if mode == 'self_test':
                A = D2*0
            elif mode == 'train':
                tmp = np.atleast_2d(x[:,der])/ell[der]
                A = sf2 * ( 1.0 + 0.5*D2/alpha )**(-alpha-1) * spdist.cdist(tmp, tmp, 'sqeuclidean')
            elif mode == 'cross':
                A = sf2 * ( 1.0 + 0.5*D2/alpha )**(-alpha-1) * spdist.cdist(np.atleast_2d(x[:,der]).T/ell[der], np.atleast_2d(z[:,der]).T/ell[der], 'sqeuclidean')
        elif der==D:                # compute derivative matrix wrt magnitude parameter
            A = 2. * sf2 * ( ( 1.0 + 0.5*D2/alpha )**(-alpha) )
        elif der==(D+1):            # compute derivative matrix wrt magnitude parameter
            K = ( 1.0 + 0.5*D2/alpha )
            A = sf2 * K**(-alpha) * ( 0.5*D2/K - alpha*np.log(K) )
        else:
            raise Exception("Wrong derivative index in covRQard") 
        return A
        


class Pre(Kernel):
    '''
    Precomputed kernel matrix. No hyperparameters and thus nothing will be optimised. 

    :param M1: cross covariances matrix(train+1 by test).
               last row is self covariances (diagonal of test by test)
    :param M2: training set covariance matrix (train by train)   
    '''
    def __init__(self,M1,M2):
        self.M1 = M1
        self.M2 = M2
        self.hyp = []

    def getCovMatrix(self,x=None,z=None,mode=None):
        if mode == 'self_test':           # diagonal covariance between test_test
            A = self.M1[-1,:]             # self covariances for the test cases (last row) 
            A = np.reshape(A, (A.shape[0],1))
        elif mode == 'train':             # covariance between train_train
            A = self.M2
        elif mode == 'cross':             # covariance between train_test
            A = self.M1[:-1,:]
        return A

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        if der != None:
            raise Exception("Error: NO optimization in precomputed kernel matrix")
        return 0
        
    

if __name__ == '__main__':
    pass




