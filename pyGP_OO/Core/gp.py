#================================================================================
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [marthaler at ge dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]
#
#    This file is part of pyGP_OO.
#    The software package is released under the BSD 2-Clause (FreeBSD) License.
#
#    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 30/09/2013
#================================================================================

import numpy as np
import matplotlib.pyplot as plt
import inf, mean, lik, cov, opt
from tools import unique
from copy import deepcopy
import pyGP_OO

#   MEANING OF NOTATION:
#  
#   inffunc      function specifying the inference method 
#   covfunc      prior covariance function (see below)
#   meanfunc     prior mean function
#   likfunc      likelihood function
#   x            n by D matrix of training inputs
#   y            column vector of length n of training targets
#   xs           ns by D matrix of test inputs
#   ys           column vector of length nn of test targets
#   der          flag for dnlZ computation determination (when xs == None also)
#
#   nlZ          returned value of the negative log marginal likelihood
#   dnlZ         column vector of partial derivatives of the negative
#                    log marginal likelihood w.r.t. each hyperparameter
#   ymu          column vector (of length ns) of predictive output means
#   ys2          column vector (of length ns) of predictive output variances
#   fmu          column vector (of length ns) of predictive latent means
#   fs2          column vector (of length ns) of predictive latent variances
#   lp           column vector (of length ns) of log predictive probabilities
#
#   post         struct representation of the (approximate) posterior
#                post is consist of post.alpha, post.L, post.sW
#
# @author: Shan Huang (last update Sep.2013)
# This is a object-oriented python implementation of gpml functionality 
# (Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18).
# based on the functional-version of python implementation
# (Copyright (c) by Marion Neumann and Daniel Marthaler, 20/05/2013)
# 
#
# Copyright (c) by Marion Neumann and Shan Huang, Sep.2013


class GP(object):
    """Base class for GP model"""
    def __init__(self):
        super(GP, self).__init__()
        self.meanfunc = None
        self.covfunc = None
        self.likfunc = None
        self.inffunc = None
        self.optimizer = None
        self._neg_log_marginal_likelihood_ = None
        self._neg_log_marginal_likelihood_gradient_ = None  
        self._posterior_ = None  
        self.x = None
        self.y = None
        self.xs = None
        self.ys = None
        self.ym = None 
        self.lp = None

    def setData(self, x, y):
        self.x = x
        self.y = y

    def plotData_1d(self, axisvals=[-1.9, 1.9, -0.9, 3.9]):
        plt.figure()
        plt.plot(self.x, self.y,' b+', markersize=12)
        plt.axis(axisvals)
        plt.grid()
        plt.xlabel('input x')
        plt.ylabel('target y')
        plt.show()
    
    def plotData_2d(self,x1,x2,t1,t2,p1,p2,axisvals=[-4, 4, -4, 4]):
        fig = plt.figure()
        plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
        plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
        pc = plt.contour(t1, t2, np.reshape(p2/(p1+p2), (t1.shape[0],t1.shape[1]) ))
        fig.colorbar(pc)
        plt.grid()
        plt.axis(axisvals)
        plt.show()

    def setPrior(self, mean=None, kernel=None):
        """set prior mean and cov"""
        if mean != None:
            self.meanfunc = mean
        if kernel != None:
            self.covfunc = kernel

    def train(self, x=None, y=None):
        '''
        train optimal hyperparameters 
        adjust to all mean/cov/lik functions
        '''
        if x != None:
            self.x = x  
        if y != None:
            self.y = y
        # optimize 
        optimalHyp, optimalNlZ = self.optimizer.findMin(self.x, self.y)
        self._neg_log_marginal_likelihood_ = optimalNlZ

        # apply optimal hyp to all mean/cov/lik functions here
        self.optimizer.apply_in_objects(optimalHyp)
        self.fit()

    def fit(self,x=None, y=None,der=True):
        '''
        fit the training data
        @return  [nlZ, post]        if der = False
        @return  [nlZ, dnlZ, post]  if der = True (default)
            
            where nlZ  is the negative log marginal likelihood
                  dnlZ is partial derivatives of nlZ w.r.t. each hyperparameter
                  post is struct representation of the (approximate) posterior
                  post is consist of post.alpha, post.L, post.sW
        '''
        if x != None:
            self.x = x  
        if y != None:
            self.y = y
        # call inference method
        if isinstance(self.likfunc, lik.Erf):  #or likLogistic)
            uy = unique(self.y)        
            ind = ( uy != 1 )
            if any( uy[ind] != -1):
                raise Exception('You attempt classification using labels different from {+1,-1}')
        if not der:
            post, nlZ = self.inffunc.proceed(self.meanfunc, self.covfunc, self.likfunc, self.x, self.y, 2)
            self._neg_log_marginal_likelihood_ = nlZ
            self._posterior_ = deepcopy(post)
            return nlZ, post          
        else:
            post, nlZ, dnlZ = self.inffunc.proceed(self.meanfunc, self.covfunc, self.likfunc, self.x, self.y, 3) 
            self._neg_log_marginal_likelihood_ = nlZ 
            self._neg_log_marginal_likelihood_gradient_ = deepcopy(dnlZ)
            self._posterior_ = deepcopy(post)
            return nlZ, dnlZ, post    

    def predict(self, xs, ys=None):
        '''
        prediction according to given inputs 

        @param xs           test input
        @param ys           test target(optional)

        @return ymu, ys2, fmu, fs2, lp
                where ymu is predictive output means
                      ys2 is predictive output variances
                      fm2 is predictive latent means
                      fs2 is predictive latent variances
                      lp  is log predictive probabilities(if ys is given, otherwise is None)
        '''
        meanfunc = self.meanfunc
        covfunc = self.covfunc
        likfunc = self.likfunc
        inffunc = self.inffunc
        x = self.x
        y = self.y
        self.xs = xs
        self.ys = ys
        
        if self._posterior_ == None:   
            self.fit()        
        alpha = self._posterior_.alpha
        L     = self._posterior_.L
        sW    = self._posterior_.sW
        
        nz = range(len(alpha[:,0]))         # non-sparse representation 
        if L == []:                         # in case L is not provided, we compute it
            K = covfunc.proceed(x[nz,:])
            L = np.linalg.cholesky( (np.eye(nz) + np.dot(sW,sW.T)*K).T )
        Ltril     = np.all( np.tril(L,-1) == 0 ) # is L an upper triangular matrix?
        ns        = xs.shape[0]                  # number of data points
        nperbatch = 1000                         # number of data points per mini batch
        nact      = 0                            # number of already processed test data points
        ymu = np.zeros((ns,1))
        ys2 = np.zeros((ns,1))
        fmu = np.zeros((ns,1))
        fs2 = np.zeros((ns,1))
        lp  = np.zeros((ns,1))
        while nact<=ns-1:                              # process minibatches of test cases to save memory
            id  = range(nact,min(nact+nperbatch,ns))   # data points to process
            kss = covfunc.proceed(xs[id,:], 'diag')    # self-variances
            Ks  = covfunc.proceed(x[nz,:], xs[id,:])   # cross-covariances
            ms  = meanfunc.proceed(xs[id,:])         
            N   = (alpha.shape)[1]                     # number of alphas (usually 1; more in case of sampling)
            Fmu = np.tile(ms,(1,N)) + np.dot(Ks.T,alpha[nz])          # conditional mean fs|f
            fmu[id] = np.reshape(Fmu.sum(axis=1)/N,(len(id),1))       # predictive means
            if Ltril: # L is triangular => use Cholesky parameters (alpha,sW,L)
                V       = np.linalg.solve(L.T,np.tile(sW,(1,len(id)))*Ks)
                fs2[id] = kss - np.array([(V*V).sum(axis=0)]).T             # predictive variances
            else:     # L is not triangular => use alternative parametrization
                fs2[id] = kss + np.array([(Ks*np.dot(L,Ks)).sum(axis=0)]).T # predictive variances
            fs2[id] = np.maximum(fs2[id],0)            # remove numerical noise i.e. negative variances
            Fs2 = np.tile(fs2[id],(1,N))               # we have multiple values in case of sampling
            if ys == None:
                [Lp, Ymu, Ys2] = likfunc.proceed(None,Fmu[:],Fs2[:],None,None,3)
            else:
                [Lp, Ymu, Ys2] = likfunc.proceed(np.tile(ys[id],(1,N)), Fmu[:], Fs2[:],None,None,3)
            lp[id]  = np.reshape( np.reshape(Lp,(np.prod(Lp.shape),N)).sum(axis=1)/N , (len(id),1) )   # log probability; sample averaging
            ymu[id] = np.reshape( np.reshape(Ymu,(np.prod(Ymu.shape),N)).sum(axis=1)/N ,(len(id),1) )  # predictive mean ys|y and ...
            ys2[id] = np.reshape( np.reshape(Ys2,(np.prod(Ys2.shape),N)).sum(axis=1)/N , (len(id),1) ) # .. variance
            nact = id[-1]+1                  # set counter to index of next data point

        self.ym = ymu
        self.ys2 = ys2
        self.lp = lp
        if ys == None:
            return ymu, ys2, fmu, fs2, None
        else:
            return ymu, ys2, fmu, fs2, lp 


    def predict_with_posterior(self, post, xs, ys=None):
        '''
        prediction with provided posterior
        (i.e. you already have the posterior and thus don't need fitting/training phases)
        
        @param post         posterior structcture
        @param xs           test input
        @param ys           test target(optional)

        @return ymu, ys2, fmu, fs2, lp
                where ymu is predictive output means
                      ys2 is predictive output variances
                      fm2 is predictive latent means
                      fs2 is predictive latent variances
                      lp  is log predictive probabilities(if ys is given, otherwise is None)
        '''
        meanfunc = self.meanfunc
        covfunc = self.covfunc
        likfunc = self.likfunc
        inffunc = self.inffunc
        x = self.x
        y = self.y
        
        self._posterior_ = deepcopy(post)
        alpha = post.alpha
        L     = post.L
        sW    = post.sW

        nz = range(len(alpha[:,0]))         # non-sparse representation 
        if L == []:                         # in case L is not provided, we compute it
            K = covfunc.proceed(x[nz,:])
            L = np.linalg.cholesky( (np.eye(nz) + np.dot(sW,sW.T)*K).T )
        Ltril     = np.all( np.tril(L,-1) == 0 ) # is L an upper triangular matrix?
        ns        = xs.shape[0]                  # number of data points
        nperbatch = 1000                         # number of data points per mini batch
        nact      = 0                            # number of already processed test data points
        ymu = np.zeros((ns,1))
        ys2 = np.zeros((ns,1))
        fmu = np.zeros((ns,1))
        fs2 = np.zeros((ns,1))
        lp  = np.zeros((ns,1))
        while nact<=ns-1:                              # process minibatches of test cases to save memory
            id  = range(nact,min(nact+nperbatch,ns))   # data points to process
            kss = covfunc.proceed(xs[id,:], 'diag')    # self-variances
            Ks  = covfunc.proceed(x[nz,:], xs[id,:])   # cross-covariances
            ms  = meanfunc.proceed(xs[id,:])         
            N   = (alpha.shape)[1]                     # number of alphas (usually 1; more in case of sampling)
            Fmu = np.tile(ms,(1,N)) + np.dot(Ks.T,alpha[nz])          # conditional mean fs|f
            fmu[id] = np.reshape(Fmu.sum(axis=1)/N,(len(id),1))       # predictive means
            if Ltril: # L is triangular => use Cholesky parameters (alpha,sW,L)
                V       = np.linalg.solve(L.T,np.tile(sW,(1,len(id)))*Ks)
                fs2[id] = kss - np.array([(V*V).sum(axis=0)]).T             # predictive variances
            else:     # L is not triangular => use alternative parametrization
                fs2[id] = kss + np.array([(Ks*np.dot(L,Ks)).sum(axis=0)]).T # predictive variances
            fs2[id] = np.maximum(fs2[id],0)            # remove numerical noise i.e. negative variances
            Fs2 = np.tile(fs2[id],(1,N))               # we have multiple values in case of sampling
            if ys == None:
                [Lp, Ymu, Ys2] = likfunc.proceed(None,Fmu[:],Fs2[:],None,None,3)
            else:
                [Lp, Ymu, Ys2] = likfunc.proceed(np.tile(ys[id],(1,N)), Fmu[:], Fs2[:],None,None,3)
            lp[id]  = np.reshape( np.reshape(Lp,(np.prod(Lp.shape),N)).sum(axis=1)/N , (len(id),1) )   # log probability; sample averaging
            ymu[id] = np.reshape( np.reshape(Ymu,(np.prod(Ymu.shape),N)).sum(axis=1)/N ,(len(id),1) )  # predictive mean ys|y and ...
            ys2[id] = np.reshape( np.reshape(Ys2,(np.prod(Ys2.shape),N)).sum(axis=1)/N , (len(id),1) ) # .. variance
            nact = id[-1]+1                  # set counter to index of next data point
        
        self.ym = ymu
        self.ys2 = ys2
        self.lp = lp
        if ys == None:
            return ymu, ys2, fmu, fs2, None
        else:
            return ymu, ys2, fmu, fs2, lp 






class GPR(GP):
    """Gaussian Process Regression"""
    def __init__(self):
        super(GPR, self).__init__()
        self.meanfunc = mean.Zero()                        # default prior mean 
        self.covfunc = cov.RBF()                           # default prior covariance
        self.likfunc = lik.Gauss()                         # likihood with default noise variance 0.1
        self.inffunc = inf.Exact()                         # inference method
        self.optimizer = opt.Minimize(self)                # default optimizer       

    def setNoise(self,log_sigma):
        """explicitly set noise variance other than default"""
        self.likfunc = lik.Gauss(log_sigma)

    def setOptimizer(self, method, num_restarts=None, min_threshold=None, meanRange=None, covRange=None, likRange=None):
        conf = None
        if (num_restarts!=None) or (min_threshold!=None):
            conf = pyGP_OO.Optimization.conf.random_init_conf(self.meanfunc,self.covfunc,self.likfunc)
            conf.num_restarts = num_restarts
            conf.min_threshold = min_threshold
            if meanRange != None:
                conf.meanRange = meanRange
            if covRange != None:
                conf.covRange = covRange
            if likRange != None:
                conf.likRange = likRange   
        if method == "Minimize":
            self.optimizer = opt.Minimize(self,conf)            
        elif method == "SCG":
            self.optimizer = opt.SCG(self,conf)  
        elif method == "CG":
            self.optimizer = opt.CG(self,conf)  
        elif method == "BFGS":
            self.optimizer = opt.BFGS(self,conf)  
                       
    def plot(self,axisvals=[-1.9, 1.9, -0.9, 3.9]):
        xs = self.xs
        x = self.x
        y = self.y
        ym = self.ym
        ys2 = self.ys2
        plt.figure()
        xss  = np.reshape(xs,(xs.shape[0],))
        ymm  = np.reshape(ym,(ym.shape[0],))
        ys22 = np.reshape(ys2,(ys2.shape[0],))
        plt.plot(xs, ym, 'g-', x, y, 'r+', linewidth = 3.0, markersize = 10.0)
        plt.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=[0.,1.0,0.0,0.8],linewidths=0.0)
        plt.grid()
        if axisvals:
            plt.axis(axisvals)
        plt.xlabel('input x')
        plt.ylabel('target y')
        plt.show()






class GPC(GP):
    """Gaussian Process Classification"""
    def __init__(self):
        super(GPC, self).__init__()
        self.meanfunc = mean.Zero()                        # default prior mean 
        self.covfunc = cov.RBF()                           # default prior covariance
        self.likfunc = lik.Erf()                           # erf likihood 
        self.inffunc = inf.EP()                            # default inference method
        self.optimizer = opt.Minimize(self)                # default optimizer       

    def useLaplace(self):
        """use Laplace approxiamation other than EP"""
        self.inffunc = inf.Laplace() 

    def setOptimizer(self, method, num_restarts=None, min_threshold=None, meanRange=None, covRange=None, likRange=None):
        conf = None
        if (num_restarts!=None) or (min_threshold!=None):
            conf = pyGP_OO.Optimization.conf.random_init_conf(self.meanfunc,self.covfunc,self.likfunc)
            conf.num_restarts = num_restarts
            conf.min_threshold = min_threshold
            if meanRange != None:
                conf.meanRange = meanRange
            if covRange != None:
                conf.covRange = covRange
            if likRange != None:
                conf.likRange = likRange   
        if method == "Minimize":
            self.optimizer = opt.Minimize(self,conf)            
        elif method == "SCG":
            self.optimizer = opt.SCG(self,conf)  
        elif method == "CG":
            self.optimizer = opt.CG(self,conf)  
        elif method == "BFGS":
            self.optimizer = opt.BFGS(self,conf)  
                       
    def plot(self,x1,x2,t1,t2,axisvals=[-4, 4, -4, 4]):
        fig = plt.figure()
        plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
        plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
        pc = plt.contour(t1, t2, np.reshape(np.exp(self.lp), (t1.shape[0],t1.shape[1]) ))
        fig.colorbar(pc)
        plt.grid()
        plt.axis(axisvals)
        plt.show()






class GPR_FITC(GP):
    """Gaussian Process Regression FITC"""
    def __init__(self):
        super(GPR_FITC, self).__init__()
        self.meanfunc = mean.Zero()                        # default prior mean 
        self.covfunc = cov.RBF()                           # default prior covariance
        self.likfunc = lik.Gauss()                         # likihood with default noise variance 0.1
        self.inffunc = inf.FITC_Exact()                    # inference method
        self.optimizer = opt.Minimize(self)                # default optimizer 
        self.u = None                                      # no default inducing points

    def setNoise(self,log_sigma):
        """explicitly set noise variance other than default"""
        self.likfunc = lik.Gauss(log_sigma)

    # override
    def setPrior(self, mean, kernel, inducing_points):
        """different from its parent method,
        prior must to be specified explicitly"""
        self.u = inducing_points
        self.meanfunc = mean
        self.covfunc = kernel.fitc(inducing_points)

    def setOptimizer(self, method, num_restarts=None, min_threshold=None, meanRange=None, covRange=None, likRange=None):
        conf = None
        if (num_restarts!=None) or (min_threshold!=None):
            conf = pyGP_OO.Optimization.conf.random_init_conf(self.meanfunc,self.covfunc,self.likfunc)
            conf.num_restarts = num_restarts
            conf.min_threshold = min_threshold
            if meanRange != None:
                conf.meanRange = meanRange
            if covRange != None:
                conf.covRange = covRange
            if likRange != None:
                conf.likRange = likRange   
        if method == "Minimize":
            self.optimizer = opt.Minimize(self,conf)            
        elif method == "SCG":
            self.optimizer = opt.SCG(self,conf)  
        elif method == "CG":
            self.optimizer = opt.CG(self,conf)  
        elif method == "BFGS":
            self.optimizer = opt.BFGS(self,conf)  
                       
    def plot(self,axisvals=[-1.9, 1.9, -0.9, 3.9]):
        plt.figure()
        xss  = np.reshape(self.xs,(self.xs.shape[0],))
        ymm  = np.reshape(self.ym,(self.ym.shape[0],))
        ys22 = np.reshape(self.ys2,(self.ys2.shape[0],))
        plt.plot(self.xs, self.ym, 'g-', self.x, self.y, 'r+', linewidth = 3.0, markersize = 10.0)
        plt.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=[0.,1.0,0.0,0.8],linewidths=0.0)
        plt.grid()
        if axisvals:
            plt.axis(axisvals)
        plt.xlabel('input x')
        plt.ylabel('output y')
        plt.plot(self.u,np.ones_like(self.u),'kx',markersize=12)
        plt.show()







class GPC_FITC(GP):
    """Gaussian Process Classification FITC"""
    def __init__(self):
        super(GPC_FITC, self).__init__()
        self.meanfunc = mean.Zero()                        # default prior mean 
        self.covfunc = cov.RBF()                           # default prior covariance
        self.likfunc = lik.Erf()                           # erf liklihood
        self.inffunc = inf.FITC_EP()                       # default inference method
        self.optimizer = opt.Minimize(self)                # default optimizer 
        self.u = None                                      # no default inducing points

    def useLaplace_FITC(self):
        """use Laplace approxiamation other than EP"""
        self.inffunc = inf.FITC_Laplace() 

    # override
    def setPrior(self, mean, kernel, inducing_points):
        """different from its parent method,
        prior must to be specified explicitly"""
        self.u = inducing_points
        self.meanfunc = mean
        self.covfunc = kernel.fitc(inducing_points)

    def setOptimizer(self, method, num_restarts=None, min_threshold=None, meanRange=None, covRange=None, likRange=None):
        conf = None
        if (num_restarts!=None) or (min_threshold!=None):
            conf = pyGP_OO.Optimization.conf.random_init_conf(self.meanfunc,self.covfunc,self.likfunc)
            conf.num_restarts = num_restarts
            conf.min_threshold = min_threshold
            if meanRange != None:
                conf.meanRange = meanRange
            if covRange != None:
                conf.covRange = covRange
            if likRange != None:
                conf.likRange = likRange   
        if method == "Minimize":
            self.optimizer = opt.Minimize(self,conf)            
        elif method == "SCG":
            self.optimizer = opt.SCG(self,conf)  
        elif method == "CG":
            self.optimizer = opt.CG(self,conf)  
        elif method == "BFGS":
            self.optimizer = opt.BFGS(self,conf)  
                       
    def plot(self,x1,x2,t1,t2,axisvals=[-4, 4, -4, 4]):
        fig = plt.figure()
        plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
        plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
        plt.plot(self.u[:,0],self.u[:,1],'ko', markersize=12)
        pc = plt.contour(t1, t2, np.reshape(np.exp(self.lp), (t1.shape[0],t1.shape[1]) ))
        fig.colorbar(pc)
        plt.grid()
        plt.axis(axisvals)
        plt.show()  






