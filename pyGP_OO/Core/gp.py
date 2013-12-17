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
import itertools
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
#   xs           n by D matrix of test inputs
#   ys           column vector of length nn of true test targets (optional)
#   der          flag for dnlZ computation determination (when xs == None also)
#
#   nlZ          returned value of the negative log marginal likelihood
#   dnlZ         column vector of partial derivatives of the negative
#                    log marginal likelihood w.r.t. each hyperparameter
#   ym           column vector (of length ns) of predictive output means
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



SHADEDCOLOR = [0.7539, 0.89453125, 0.62890625, 1.0]
MEANCOLOR = [ 0.2109375, 0.63385, 0.1796875, 1.0]
DATACOLOR = [0.12109375, 0.46875, 1., 1.0]

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
        self.ys2 = None
        self.lp = None

    def setData(self, x, y):
        self.x = x
        self.y = y

    def plotData_1d(self, axisvals=[-1.9, 1.9, -0.9, 3.9]):
        plt.figure()
        plt.plot(self.x, self.y, ls='None', marker='+', color=DATACOLOR, ms=12, mew=2)
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
    
    def setOptimizer(self, method, num_restarts=None, min_threshold=None, meanRange=None, covRange=None, likRange=None):
        '''This method is used to sepecify optimization configuration. By default, gp uses a single run "minimize".

        :param method: Optimization methods. Possible values are:

                        "Minimize"   -> minimize by Carl Rasmussen (python implementation of "minimize" in GPML)

                        "CG"         -> conjugent gradient

                        "BFGS"       -> quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)

                        "SCG"        -> scaled conjugent gradient (faster than CG) 
        :param num_restarts: Set if you want to run mulitiple times of optimization with different initial guess. 

                             It specifys the maximum number of runs/restarts/trials.
        :param min_threshold: Set if you want to run mulitiple times of optimization with different initial guess. 

                              It specifys the threshold of objective function value. Stop optimization when this value is reached.
        :param meanRange: The range of initial guess for mean hyperparameters. 

                          e.g. meanRange = [(-2,2), (-5,5), (0,1)]

                          Each tuple specifys the range (low, high) of this hyperparameter,

                          This is only the range of initial guess, during optimization process, optimal hyperparameters may go out of this range.

                          (-5,5) for each hyperparameter by default.
        :param covRange: The range of initial guess for kernel hyperparameters. Usage see meanRange
        :param likRange: The range of initial guess for likelihood hyperparameters. Usage see meanRange

        '''
        pass

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
        plt.plot(x, y, color=DATACOLOR, ls='None', marker='+',ms=12, mew=2)
        plt.plot(xs, ym, color=MEANCOLOR, ls='-', lw=3.)

        plt.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=SHADEDCOLOR,linewidths=0.0)
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




class GPMC(object):
    """This is a one vs. one classification wrapper for GPC"""
    def __init__(self, n_class):
        self.meanfunc = mean.Zero()                # default prior mean 
        self.covfunc = cov.RBF()                   # default prior covariance
        self.n_class = n_class                     # number of different classes
        self.x_all = None
        self.y_all = None
        self.Laplace = False
        self.newPrior = False

    def useLaplace(self):
        """use Laplace approxiamation other than EP"""
        self.Laplace = True

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
    

    # for multi-class, data is x_all and y_all
    def setData(self,x,y):
        self.x_all = x
        self.y_all = y


    def fitAndPredict(self, xs):
        '''
        predictive_vote is a matrix where
        row i    -> each test point i
        column j -> probability for being eahc class j
        '''
        predictive_vote = np.zeros((xs.shape[0],self.n_class))
        for i in xrange(self.n_class):         # classifier for class i...
            for j in xrange(i+1,self.n_class): # ...and class j
                x,y = self.createBinaryClass(i,j)
                model = GPC()
                if self.newPrior:
                    model.setPrior(mean=self.meanfunc, kernel=self.covfunc)
                if self.Laplace:
                    model.useLaplace()
                model.fit(x,y)               # fitting
                ym = model.predict(xs)[0]
                ym += 1     # now scale into 0 to 2,  ym=0 is class j, ym=2 is class i 
                vote_i = np.zeros((xs.shape[0],self.n_class))
                vote_j = np.zeros((xs.shape[0],self.n_class))
                vote_i[:,i:i+1] = ym
                vote_j[:,j:j+1] = 2-ym
                predictive_vote += vote_i
                predictive_vote += vote_j
        predictive_vote /=  predictive_vote.sum(axis=1)[:,np.newaxis]
        return predictive_vote


    def trainAndPredict(self, xs):
        '''
        predictive_vote is a matrix where
        row i    -> each test point i
        column j -> probability for being eahc class j

        '''
        predictive_vote = np.zeros((xs.shape[0],self.n_class))
        for i in xrange(self.n_class):         # classifier for class i...
            for j in xrange(i+1,self.n_class): # ...and class j
                x,y = self.createBinaryClass(i,j)
                model = GPC()
                if self.newPrior:
                    model.setPrior(mean=self.meanfunc, kernel=self.covfunc)
                if self.Laplace:
                    model.useLaplace()
                model.train(x,y)               # training
                ym = model.predict(xs)[0]
                ym += 1     # now scale into 0 to 2,  ym=0 is class j, ym=2 is class i 
                vote_i = np.zeros((xs.shape[0],self.n_class))
                vote_j = np.zeros((xs.shape[0],self.n_class))
                vote_i[:,i:i+1] = ym
                vote_j[:,j:j+1] = 2-ym
                predictive_vote += vote_i
                predictive_vote += vote_j
        predictive_vote /=  predictive_vote.sum(axis=1)[:,np.newaxis]
        return predictive_vote



    def setPrior(self, mean=None, kernel=None):
        """set prior mean and cov"""
        if mean != None:
            self.meanfunc = mean
        if kernel != None:
            self.covfunc = kernel
        self.newPrior = True

    def createBinaryClass(self, i,j):
        ''' create data points x,y which only contains class i and j
        class_i is +1 and class_j is -1'''
        class_i = []
        class_j = []
        for index in xrange(len(self.y_all)):       # check all classes
            target = self.y_all[index]
            if target == i:
                class_i.append(index)
            elif target == j:
                class_j.append(index)
        n1 = len(class_i)
        n2 = len(class_j)
        class_i.extend(class_j)
        x = self.x_all[class_i,:]
        y = np.concatenate((np.ones((1,n1)),-np.ones((1,n2))),axis=1).T
        return x,y







class GP_FITC(GP):
    """GP_FITC base class"""
    def __init__(self):
        super(GP_FITC, self).__init__()
        self.u = None

    # override
    def setData(self, x, y, value_per_axis=5):
        '''set Data and derive deault inducing_points

        value_per_axis is number of value in each dimension...
        ...when using a default inducing point grid'''
        self.x = x
        self.y = y
        # get range of x in each dimension
        # 5 uniformally selected value for each dimension
        gridAxis=[]
        for d in xrange(x.shape[1]):
            column = x[:,d]
            mini = np.min(column)
            maxi = np.max(column)
            axis = np.linspace(mini,maxi,value_per_axis)
            gridAxis.append(axis)
        # default inducing points-> a grid
        if self.u == None:
            self.u = np.array(list(itertools.product(*gridAxis)))
            self.covfunc = self.covfunc.fitc(self.u)

    # override
    def setPrior(self, mean=None, kernel=None, inducing_points=None):
        """set prior and inducing_points"""
        if kernel != None:
            if inducing_points != None:
                self.covfunc = kernel.fitc(inducing_points)
                self.u = inducing_points
            else:
                if self.u != None:
                    self.covfunc = kernel.fitc(self.u)
                else:
                    raise error("To use default inducing points, please call setData() first!")
        if mean != None:
            self.meanfunc = mean


       




class GPR_FITC(GP_FITC):
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
        plt.plot(self.x, self.y, color=DATACOLOR, ls='None', marker='+',ms=12, mew=2)
        plt.plot(self.xs, self.ym, color=MEANCOLOR, ls='-', lw=3.)
        plt.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=SHADEDCOLOR,linewidths=0.0)
        plt.grid()
        if axisvals:
            plt.axis(axisvals)
        plt.xlabel('input x')
        plt.ylabel('output y')
        plt.plot(self.u,np.ones_like(self.u), ls='None', color='k',marker='x',markersize=12,mew=2)
        plt.show()

   
   
class GPC_FITC(GP_FITC):
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






