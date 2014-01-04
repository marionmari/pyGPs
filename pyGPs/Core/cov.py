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

# @author: Shan Huang (last update Sep.2013)
# This is a object-oriented python implementation of gpml functionality 
# (Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18).
# based on the functional-version of python implementation
# (Copyright (c) by Marion Neumann and Daniel Marthaler, 20/05/2013)
# 
# Copyright (c) by Marion Neumann and Shan Huang, 30/092013


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
    def proceed(self):
        pass
    #overloading operators
    def __add__(self,cov):
        return SumOfKernel(self,cov)
    def __mul__(self,other):
        # using * for both scalar and production
        # depending on the types of two objects.
        if isinstance(other, int) or isinstance(other, float):
            return ScaleOfKernel(self,other)
        elif isinstance(other, Kernel):
            return ProductOfKernel(self,other)
        else:
            print "only numbers and Kernels are supported operand types for *"
    # FITC approximation
    def fitc(self,inducingInput):
        return FITCOfKernel(self,inducingInput)
    
    # not used anymore
    # replaced by spdist from scipy
    def sq_dist(self, a, b=None):
        # Compute a matrix of all pairwise squared distances
        # between two sets of vectors, stored in the row of the two matrices:
        # a (of size n by D) and b (of size m by D). 
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

    def proceed(self,x=None,z=None,der=None):
        n, D = x.shape
        AT = self.cov1.proceed(x,z)
        A = np.ones_like(AT)   
        if der == None:                          # compute cov vector
            A *= self.cov1.proceed(x,z)
            A *= self.cov2.proceed(x,z)
        elif isinstance(der, int):               # compute derivative vector  
            if der < len(self.cov1.hyp):
                A *= self.cov1.proceed(x, z, der)
                A *= self.cov2.proceed(x, z)
            elif der < len(self.hyp):
                der2 = der - len(self.cov1.hyp)
                A *= self.cov2.proceed(x, z, der2)
                A *= self.cov1.proceed(x,z) 
            else:
                raise Exception("Error: der out of range for covProduct")            
        return A
        

class SumOfKernel(Kernel):
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

    def proceed(self,x=None,z=None,der=None):
        n, D = x.shape
        AT = self.cov1.proceed(x,z)
        A = np.zeros_like(AT)                   # Allocate covariance Matrix
        if der == None:                         # compute cov vector
            A += self.cov1.proceed(x,z)
            A += self.cov2.proceed(x,z)
        elif isinstance(der, int):              # compute derivative vector  
            if der < len(self.cov1.hyp):
                A += self.cov1.proceed(x, z, der)
            elif der < len(self.hyp):
                der2 = der - len(self.cov1.hyp)
                A += self.cov2.proceed(x, z, der2)
            else:
                raise Exception("Error: der out of range for covSum")            
        return A
    

class ScaleOfKernel(Kernel):
# Compose a covariance function as a scaled version of another one
# k(x^p,x^q) = sf2 * k0(x^p,x^q)
#
# The hyperparameter is :
# hyp = [ log(sf2) ]
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

    def proceed(self,x=None,z=None,der=None):
        sf2 = np.exp(2.* self.hyp[0])                 # scale parameter   
        if der == None:                               # compute cov vector
            A = sf2 * self.cov.proceed(x,z)           # accumulate cov
        elif isinstance(der, int) and der == 0:       # compute derivative w.r.t. sf2
            A = 2. * sf2 * self.cov.proceed(x,z)
        else:                                 
            A = sf2 * self.cov.proceed(x, z, der-1) 
        return A
     

class FITCOfKernel(Kernel):
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

    def proceed(self,x=None,z=None,der=None):
    # Covariance function to be used together with the FITC approximation.
    # The function allows for more than one output argument and does not respect the
    # interface of a proper covariance function. 
    # Instead of outputing the full covariance, it returns cross-covariances between
    # the inputs x, z and the inducing inputs xu as needed by infFITC
        xu = self.inducingInput
        try:
            assert(xu.shape[1] == x.shape[1])
        except AssertionError:
            raise Exception('Dimensionality of inducing inputs must match training inputs')        
        if der == None:                        # compute covariance matrices for dataset x
            if z == None:
                K   = self.covfunc.proceed(x,'diag')
                Kuu = self.covfunc.proceed(xu)
                Ku  = self.covfunc.proceed(xu,x)
            elif z == 'diag':
                K = self.covfunc.proceed(x,z)
                return K
            else:
                K = self.covfunc.proceed(xu,z)
                return K
        else:                                  # compute derivative matrices
            if z == None:
                K   = self.covfunc.proceed(x,'diag',der)
                Kuu = self.covfunc.proceed(xu,None,der)
                Ku  = self.covfunc.proceed(xu,x,der)
            elif z == 'diag':
                K = self.covfunc.proceed(x,z,der)
                return K
            else:
                K = self.covfunc.proceed(xu,z,der)
                return K
        return K, Kuu, Ku


class Poly(Kernel):
    '''
    Polynomial covariance function. hyp = [ log_c, log_sigma ]

    :param log_c: inhomogeneous offset. 
    :param log_sigma: signal deviation. 
    :param log_d: order of polynomial (treated not as hyperparameter, i.e. will not be trained). 
    '''
    def __init__(self, log_c=0., log_d=np.log(2), log_sigma=0. ):
        self.hyp = [ log_c, log_sigma]
        self.para =  [log_d] 

    def proceed(self, x=None, z=None, der=None):
        c   = np.exp(self.hyp[0])             # inhomogeneous offset
        sf2 = np.exp(2.*self.hyp[1])          # signal variance
        ord = np.exp(self.para[0])            # order of polynomial
        if np.abs(ord-np.round(ord)) < 1e-8:  # remove numerical error from format of parameter
            ord = int(round(ord))
        assert(ord >= 1.)                     # only nonzero integers for ord
        ord = int(ord)       
        n,D = x.shape
        if z == 'diag':
            A = np.reshape(np.sum(x*x,1), (n,1))
        elif z == None:
            A = np.dot(x,x.T)
        else:                              # compute covariance between data sets x and z
            A = np.dot(x,z.T)              # cross covariances
    
        if der == None:                    # compute covariance matix for dataset x
            A = sf2 * (c + A)**ord
        else:
            if der == 0:      			# compute derivative matrix wrt 1st parameter             
                A = c * ord * sf2 * (c+A)**(ord-1)
            elif der == 1:  			# compute derivative matrix wrt 2nd parameter
                A = 2. * sf2 * (c + A)**ord
            elif der == 2:  			# no derivative wrt 3rd parameter
                A = np.zeros_like(A)        	# do nothing (d is not learned)
            else:
                raise Exception("Wrong derivative entry in covPoly")
        return A
   


class PiecePoly(Kernel):
    '''
    Piecewise polynomial kernel with compact support.
    hyp = [log_ell, log_sigma, log_v]

    :param log_ell: characteristic length scale. 
    :param log_sigma: signal deviation. 
    :param log_v: degree in piecewise polynomial kernel. v will be rounded to 0,1,2,or 3.
    '''
    def __init__(self, log_c=0., log_d=np.log(2), log_sigma=0. ):
        self.hyp = [log_ell, log_sigma, log_v]

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
             raise Exception (["Wrong degree in covPPiso.  Should be 0,1,2 or 3, is " + str(v)])

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
            raise Exception (["Wrong degree in covPPiso.  Should be 0,1,2 or 3, is " + str(v)])

    def pp(self,r,j,v,func):
        return func(v,r,j)*(ppmax(1-r,0)**(j+v))

    def dpp(self,r,j,v,func,dfunc):
        return ppmax(1-r,0)**(j+v-1) * r * ( (j+v)*func(v,r,j) - ppmax(1-r,0) * dfunc(v,r,j) )

    def proceed(self, x=None, z=None, der=None):
        ell = np.exp(self.hyp[0])            # characteristic length scale
        sf2 = np.exp(2.*self.hyp[1])         # signal variance
        v   = np.exp(self.hyp[2])            # degree (v = 0,1,2 or 3 only)
        if np.abs(v-np.round(v)) < 1e-8:     # remove numerical error from format of parameter
            v = int(round(v))
        assert(int(v) in range(4))           # Only allowed degrees: 0,1,2 or 3
        v = int(v)        
        n, D = x.shape
        j = np.floor(0.5*D) + v + 1
        if z == 'diag':
            A = np.zeros((n,1))
        elif z == None:
            A = np.sqrt( spdist.cdist(x/ell, x/ell, 'sqeuclidean') )
        else:                                       # compute covariance between data sets x and z
            A = np.sqrt( spdist.cdist(x/ell, z/ell, 'sqeuclidean') )     # cross covariances 
        if der == None:                             # compute covariance matix for dataset x
            A = sf2 * pp(A,j,v,func)
        else:
            if der == 0:                            # compute derivative matrix wrt 1st parameter
                A = sf2 * dpp(A,j,v,func,dfunc)

            elif der == 1:                          # compute derivative matrix wrt 2nd parameter
                A = 2. * sf2 * pp(A,j,v,func)

            elif der == 2:                          # wants to compute derivative wrt order
                A = np.zeros_like(A)
            else:
                raise Exception("Wrong derivative entry in covPPiso")
        return A



class RBF(Kernel):
    '''
    Squared Exponential kernel with isotropic distance measure. hyp = [log_ell, log_sigma]

    :param log_ell: characteristic length scale. 
    :param log_sigma: signal deviation. 
    '''
    def __init__(self, log_ell=-1., log_sigma=0.):
        self.hyp = [log_ell, log_sigma]

    def proceed(self, x=None, z=None, der=None):

        ell = np.exp(self.hyp[0])         # characteristic length scale
        sf2 = np.exp(2.*self.hyp[1])      # signal variance
        n,D = x.shape
        if z == 'diag':
            A = np.zeros((n,1))
        elif z == None:
            A = spdist.cdist(x/ell,x/ell,'sqeuclidean')
        else:                              # compute covariance between data sets x and z
            A = spdist.cdist(x/ell,z/ell,'sqeuclidean') # self covariances      
        if der == None:                    # compute covariance matix for dataset x
            A = sf2 * np.exp(-0.5*A)
        else:
            if der == 0:    # compute derivative matrix wrt 1st parameter
                A = sf2 * np.exp(-0.5*A) * A
            elif der == 1:  # compute derivative matrix wrt 2nd parameter
                A = 2. * sf2 * np.exp(-0.5*A)
            else:
                raise Exception("Calling for a derivative in covSEiso that does not exist")
        return A


class RBFunit(Kernel):
    '''
    Squared Exponential kernel with isotropic distance measure with unit magnitude.
    i.e signal variance is always 1. hyp = [ log_ell ]

    :param log_ell: characteristic length scale. 
    '''
    def __init__(self, log_ell=-1.):

        self.hyp = [log_ell]

    def proceed(self, x=None, z=None, der=None):
        ell = np.exp(self.hyp[0])  # characteristic length scale
        n,D = x.shape
        if z == 'diag':
            A = np.zeros((n,1))
        elif z == None:
            A = spdist.cdist(x/ell, x/ell, 'sqeuclidean')
        else:                      # compute covariance between data sets x and z
            A = spdist.cdist(x/ell, z/ell, 'sqeuclidean')   # self covariances (needed for GPR)
        if der == None:            # compute covariance matix for dataset x
            A = np.exp(-0.5*A)
        else:
            if der == 0:           # compute derivative matrix wrt 1st parameter
                A = np.exp(-0.5*A) * A
            else:
                raise Exception("Wrong derivative index in covSEisoU")
        return A


class RBFard(Kernel):
    '''
    Squared Exponential kernel with Automatic Relevance Determination.
    hyp = log_ell_list + [log_sigma]

    :param D: dimension of pattern. set if you want default ell, which is 0.5 for each dimension.
    :param log_ell_list: characteristic length scale for each dimension.
    :param log_sigma: signal deviation. 
    '''
    def __init__(self, D=None, log_ell_list=None, log_sigma=0.):
        if log_ell_list == None:
            self.hyp = [0.5 for i in xrange(D)] + [log_sigma]
        else:
            self.hyp = log_ell_list + [log_sigma]
    def proceed(self, x=None, z=None, der=None):
        n, D = x.shape  
        ell = 1./np.exp(self.hyp[0:D])    # characteristic length scale
        sf2 = np.exp(2.*self.hyp[D])      # signal variance
        if z == 'diag':
            A = np.zeros((n,1))
        elif z == None:
            tem = np.dot(np.diag(ell),x.T).T
            A = spdist.cdist(tem,tem,'sqeuclidean')
        else:                # compute covariance between data sets x and z
            A = spdist.cdist(np.dot(np.diag(ell),x.T).T,np.dot(np.diag(ell),z.T).T,'sqeuclidean')
        A = sf2*np.exp(-0.5*A)
        if der:
            if der < D:      # compute derivative matrix wrt length scale parameters
                if z == 'diag':
                    A = A*0
                elif z == None:
                    tem = np.atleast_2d(x[:,der])/ell[der]
                    A *= spdist.cdist(tem,tem,'sqeuclidean')
                else:
                    A *= spdist.cdist(np.atleast_2d(x[:,der]).T/ell[der],np.atleast_2d(z[:,der]).T/ell[der],'sqeuclidean')
            elif der==D:     # compute derivative matrix wrt magnitude parameter
                A = 2.*A
            else:
                raise Exception("Wrong derivative index in covSEard")   
        return A

            



class Const(Kernel):
    '''
    Constant kernel. hyp = [ log_sigma ]

    :param log_sigma: signal deviation. 
    '''
    def __init__(self, log_sigma=0.):
        self.hyp = [log_sigma]

    def proceed(self, x=None, z=None, der=None):
        sf2 = np.exp(self.hyp[0])         # s2
        n,m = x.shape
        if z == 'diag':
            A = sf2*np.ones((n,1))
        elif z == None:
            A = sf2 * np.ones((n,n))
        else:
            A = sf2*np.ones((n,z.shape[0]))
        if der == 0:  # compute derivative matrix wrt sf2
            A = 2. * A
        elif der:
            raise Exception("Wrong derivative entry in covConst")
        return A


class LIN(Kernel):
    '''
    Linear kernel. No hyperparameters.
    '''
    def __init__(self):
        self.hyp = []
    def proceed(self, x=None, z=None, der=None):
        n,m = x.shape
        if z == 'diag':
            A = np.reshape(np.sum(x*x,1), (n,1))
        elif z == None:
            A = np.dot(x,x.T) + np.eye(n)*1e-16     # required for numerical accuracy
        else:                                       # compute covariance between data sets x and z
            A = np.dot(x,z.T)                       # cross covariances
        if der:
            raise Exception("No derivative available in covLIN")
        return A


class LINard(Kernel):
    '''
    Linear covariance function with Automatic Relevance Detemination.
    hyp = log_ell_list 

    :param D: dimension of training data. Set if you want default ell, which is 0.5 for each dimension.
    :param log_ell_list: characteristic length scale for each dimension.
    '''

    def __init__(self, D=None, log_ell_list=None):
        if log_ell_list == None:
            self.hyp = [0.5 for i in xrange(D)]
        else:
            self.hyp = log_ell_list

    def proceed(self, x=None, z=None, der=None):
        n, D = x.shape
        ell = np.exp(self.hyp)            # ARD parameters
        x = np.dot(x,np.diag(1./ell))
        if z == 'diag':
            A = np.reshape(np.sum(x*x,1), (n,1))
        elif z == None:
            A = np.dot(x,x.T)
        else:                             # compute covariance between data sets x and z
            z = np.dot(z,np.diag(1./ell))
            A = np.dot(x,z.T)             # cross covariances

        if not der == None and der < D:
            if z == 'diag':
                A = -2.*x[:,der]*x[:,der]
            elif z == None:
                A = -2.*np.dot(x[:,der],x[:,der].T)
            else:
                A = -2.*np.dot(x[:,der],z[:,der].T) # cross covariances
        elif der:
            raise Exception("Wrong derivative index in covLINard")
        return A


class Matern(Kernel):
    '''
    Matern covariance function with nu = d/2 and isotropic distance measure. 
    For d=1 the function is also known as the exponential covariance function 
    or the Ornstein-Uhlenbeck covariance in 1d.
    d will be rounded to 1, 3, or 5.

    hyp = [ log_ell, log_sigma, log_d ]
    
    :param log_d: d is 2 times nu
    :param log_ell: characteristic length scale. 
    :param log_sigma: signal deviation. 
    '''
    def __init__(self, log_ell=-1., log_d=0., log_sigma=0. ):
        self.hyp = [ log_ell, log_sigma, log_d ]

    def func(self,d,t):
        if d == 1:
            return 1
        elif d == 3:
            return 1 + t
        elif d == 5:
            return 1 + t*(1+t/3.)
        else:
            raise Exception("Wrong value for d in covMatern")
    def dfunc(self,d,t):
        if d == 1:
            return 1
        elif d == 3:
            return t
        elif d == 5:
            return t*(1+t/3.)
        else:
            raise Exception("Wrong value for d in covMatern")
    def mfunc(self,d,t):
        return self.func(d,t)*np.exp(-1.*t)
    def dmfunc(self,d,t):
        return self.dfunc(d,t)*t*np.exp(-1.*t)
    def proceed(self, x=None, z=None, der=None):
        ell = np.exp(self.hyp[0])        # characteristic length scale
        sf2 = np.exp(2.* self.hyp[1])    # signal variance
        d   = np.exp(self.hyp[2])        # 2 times nu
        if np.abs(d-np.round(d)) < 1e-8: # remove numerical error from format of parameter
            d = int(round(d))
        d = int(d)
        try:
            assert(d in [1,3,5])         # check for valid values of d
        except AssertionError:
            d = 3

        if z == 'diag':
            A = np.zeros((x.shape[0],1))
        elif z == None:
            x = np.sqrt(d)*x/ell   
            A = np.sqrt(spdist.cdist(x, x, 'sqeuclidean'))
        else:
            x = np.sqrt(d)*x/ell
            z = np.sqrt(d)*z/ell
            A = np.sqrt(spdist.cdist(x, z, 'sqeuclidean'))

        if der == None:                     # compute covariance matix for dataset x
            A = sf2 * self.mfunc(d,A)
        else:
            if der == 0:                    # compute derivative matrix wrt 1st parameter
                A = sf2 * self.dmfunc(d,A)
            elif der == 1:                  # compute derivative matrix wrt 2nd parameter
                A = 2 * sf2 * self.mfunc(d,A)
            elif der == 2:                  # no derivative wrt 3rd parameter
                A = np.zeros_like(A)        # do nothing (d is not learned)
            else:
                raise Exception("Wrong derivative value in covMatern")
        return A



class Periodic(Kernel):
    '''
    Stationary kernel for a smooth periodic function. 
    hyp = [ log_ell, log_p, log_sigma]

    :param log_p: period.
    :param log_ell: characteristic length scale. 
    :param log_sigma: signal deviation. 
    '''
    def __init__(self, log_ell=-1, log_p=0., log_sigma=0. ): 
        self.hyp = [ log_ell, log_p, log_sigma]

    def proceed(self, x=None, z=None, der=None):
        ell = np.exp(self.hyp[0])        # characteristic length scale
        p   = np.exp(self.hyp[1])        # period
        sf2 = np.exp(2.*self.hyp[2])     # signal variance
        n,D = x.shape
        if z == 'diag':
            A = np.zeros((n,1))
        elif z == None:
            A = np.sqrt(spdist.cdist(x, x, 'sqeuclidean'))
        else:
            A = np.sqrt(spdist.cdist(x, z, 'sqeuclidean'))
        A = np.pi*A/p
        if der == None:             # compute covariance matix for dataset x
            A = np.sin(A)/ell
            A = A * A
            A = sf2 *np.exp(-2.*A)
        else:
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

    def proceed(self, x=None, z=None, der=None):
        tol = 1.e-9                 # Tolerance for declaring two vectors "equal"
        s2 = np.exp(2.*self.hyp[0]) # noise variance
        n,D = x.shape
        if z == 'diag':
            A = np.ones((n,1))
        elif z == None:
            A = np.eye(n)
        else:                       # compute covariance between data sets x and z
            M = spdist.cdist(x, z, 'sqeuclidean')
            A = np.zeros_like(M,dtype=np.float)
            A[M < tol] = 1.
        if der == None:
            A = s2*A
        else:                       # compute derivative matrix
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
    :param log_alpha: hape parameter for the RQ covariance.
    '''
    def __init__(self, log_ell=-1, log_sigma=0., log_alpha=0.):
        self.hyp = [ log_ell, log_sigma, log_alpha ]

    def proceed(self, x=None, z=None, der=None):
        ell   = np.exp(self.hyp[0])            # characteristic length scale
        sf2   = np.exp(2.*self.hyp[1])         # signal variance
        alpha = np.exp(self.hyp[2])            
        n,D = x.shape
        if z == 'diag':
            D2 = np.zeros((n,1))
        elif z == None:
            D2 = spdist.cdist(x/ell, x/ell, 'sqeuclidean')
        else:
            D2 = spdist.cdist(x/ell, z/ell, 'sqeuclidean')
        if der == None:                  # compute covariance matix for dataset x
            A = sf2 * ( ( 1.0 + 0.5*D2/alpha )**(-alpha) )
        else:
            if der == 0:                # compute derivative matrix wrt 1st parameter
                A = sf2 * ( 1.0 + 0.5*D2/alpha )**(-alpha-1) * D2

            elif der == 1:              # compute derivative matrix wrt 2nd parameter
                A = 2.* sf2 * ( ( 1.0 + 0.5*D2/alpha )**(-alpha) )

            elif der == 2:              # compute derivative matrix wrt 3rd parameter
                K = ( 1.0 + 0.5*D2/alpha )
                A = sf2 * K**(-alpha) * (0.5*D2/K - alpha*np.log(K) )
            else:
                raise Exception("Wrong derivative index in covRQiso")
        return A



class RQard(Kernel):
    '''
    Rational Quadratic covariance function with Automatic Relevance Detemination
     (ARD) distance measure.
    hyp = log_ell_list + [ log_sigma, log_alpha ]

    :param D: dimension of pattern. set if you want default ell, which is 0.5 for each dimension.
    :param log_ell_list: characteristic length scale for each dimension.
    :param log_sigma: signal deviation. 
    :param log_alpha: hape parameter for the RQ covariance.
    '''
    def __init__(self, D=None, log_ell_list=None, log_sigma=0., log_alpha=0.):
        if log_ell_list == None:
            self.hyp = [0.5 for i in xrange(D)] + [ log_sigma, log_alpha ]
        else:
            self.hyp = log_ell_list + [ log_sigma, log_alpha ]

    def proceed(self, x=None, z=None, der=None):
        n, D = x.shape  
        ell = 1./np.exp(self.hyp[0:D])    # characteristic length scale
        sf2 = np.exp(2.*self.hyp[D])      # signal variance
        alpha = np.exp(self.hyp[D+1])
        if z == 'diag':
            D2 = np.zeros((n,1))
        elif z == None:
            tmp = np.dot(np.diag(ell),x.T).T
            D2 = spdist.cdist(tmp, tmp, 'sqeuclidean')
        else:
            D2 = spdist.cdist(np.dot(np.diag(ell),x.T).T, np.dot(np.diag(ell),z.T).T, 'sqeuclidean')
        if der == None:                 # compute covariance matix for dataset x
            A = sf2 * ( ( 1.0 + 0.5*D2/alpha )**(-alpha) )
        else:
            if der < D:                 # compute derivative matrix wrt length scale parameters
                if z == 'diag':
                    A = D2*0
                elif z == None:
                    tmp = np.atleast_2d(x[:,der])/ell[der]
                    A = sf2 * ( 1.0 + 0.5*D2/alpha )**(-alpha-1) * spdist.cdist(tmp, tmp, 'sqeuclidean')
                else:
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
    def proceed(self, x=None, z=None, der=None):
        if z == 'diag':             # diagonal covariance between test_test
            A = self.M1[-1,:]            # self covariances for the test cases (last row) 
            A = np.reshape(A, (A.shape[0],1))
        elif z == None:             # covariance between train_train
            A = self.M2
        else:                       # covariance between train_test
            A = self.M1[:-1,:]
        if der != None:
            raise Exception("Error: NO optimization in precomputed kernel matrix")
        return A

    

    
    

    

if __name__ == '__main__':
    # test1: combinations of kernel functions
    k1 = covPoly([1,2,3])
    k2 = covPoly([2,2,2])
    k3 = covPoly([4,4,4])
    myCov = k1*k2*6 + k1*k2
    print myCov.hyp

    #########################################

    # test2: Does proceed() perform correctly compare to feval()?
    n = 20 # number of labeled/training data
    D = 1  # Dimension of input data
    x = np.array([2.083970427750732,  -0.821018066101379,  -0.617870699182597,  -1.183822608860694,\
              0.274087442277144,   0.599441729295593,   1.768897919204435,  -0.465645549031928,\
              0.588852784375935,  -0.832982214438054,  -0.512106527960363,   0.277883144210116,\
              -0.065870426922211,  -0.821412363806325,   0.185399443778088,  -0.858296174995998,\
               0.370786630037059,  -1.409869162416639,-0.144668412325022,-0.553299615220374]);
    x = np.reshape(x,(n,D))
    z = np.array([np.linspace(-1.9,1.9,101)]).T
    
    k = covSEard([0.1,0.2])
    print myCov.proceed(x,z,1)

    # have the same result if passing same inputs to feval() 
    # covPoly tested!




