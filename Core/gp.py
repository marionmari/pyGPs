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
import inf, mean, lik, cov, opt
from tools import unique

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

    def withData(self, x, y):
        self.x = x
        self.y = y
 

    def withPrior(self, mean=None, kernel=None):
        """set prior mean and cov"""
        self.meanfunc = mean
        self.covfunc = kernel


    def train(self):
        '''
        train optimal hyperparameters 
        adjust to all mean/cov/lik functions
        '''
        meanfunc = self.meanfunc
        covfunc = self.covfunc
        likfunc = self.likfunc
        inffunc = self.inffunc
        optimizer = self.optimizer

        # optimize here
        optimalHyp, optimalNlZ = optimizer.findMin(self.x, self.y)
        self._neg_log_marginal_likelihood_ = optimalNlZ

        # apply optimal hyp to all mean/cov/lik functions here
        optimizer.apply_in_objects(optimalHyp)


    def fit(self, der=True):
        '''
        fit the training data
        @return  [nlZ, post]        if der = False
        @return  [nlZ, dnlZ, post]  if der = True (default)
            
            where nlZ  is the negative log marginal likelihood
                  dnlZ is partial derivatives of nlZ w.r.t. each hyperparameter
                  post is struct representation of the (approximate) posterior
                  post is consist of post.alpha, post.L, post.sW
        '''
        meanfunc = self.meanfunc
        covfunc = self.covfunc
        likfunc = self.likfunc
        inffunc = self.inffunc

        # call inference method
        if isinstance(likfunc, lik.likErf):  #or likLogistic)
            uy = unique(y)        
            ind = ( uy != 1 )
            if any( uy[ind] != -1):
                raise Exception('You attempt classification using labels different from {+1,-1}')
        if not der:
            post, nlZ = inffunc.proceed(meanfunc, covfunc, likfunc, self.x, self.y, 2)
            self._neg_log_marginal_likelihood_ = nlZ
            self._posterior_ = post
            return nlZ, post          
        else:
            post, nlZ, dnlZ = inffunc.proceed(meanfunc, covfunc, likfunc, self.x, self.y, 3) 
            self._neg_log_marginal_likelihood_ = nlZ 
            self._neg_log_marginal_likelihood_gradient_ = dnlZ
            self._posterior_ = post
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
        
        if self._posterior_ == None:         # if posterior is not calculated before...
            self.fit(der=False)              # ...first fit training data
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
        
        self._posterior_ = post         
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
        if ys == None:
            return ymu, ys2, fmu, fs2, None
        else:
            return ymu, ys2, fmu, fs2, lp 






class GPR(GP):
    """Gaussian Process Regression"""
    def __init__(self):
        super(GPR, self).__init__()
        self.meanfunc = mean.meanZero()                        # default prior mean 
        self.covfunc = cov.rbf(lengthscale=1.0, variance=0.1)  # default prior covariance
        self.likfunc = lik.likGauss(0.1)                       # likihood with default noise variance 0.1
        self.inffunc = inf.infExact()                          # inference method
        self.optimizer = opt.Minimize(self)                    # default optimizer

    def hasNoise(self, noise_variance):
        """explicitly set noise variance other than default"""
        self.likfunc = lik.likGauss(noise_variance)









        

"""
def predict(inffunc, meanfunc, covfunc, likfunc, x, y, xs, ys=None):
    '''
    prediction according to given inputs
    return [ymu, ys2, fmu, fs2, lp] 

    If given ys, lp will be calculated.
    Otherwise,   lp is None.

    If you don't know posterior yet, you can pass y istead of post.
    '''
    if not meanfunc:
        meanfunc = mean.meanZero()  
    if not covfunc:
        raise Exception('Covariance function cannot be empty')
    if not likfunc:
        likfunc = lik.likGauss([0.1])  
    if not inffunc:
        inffunc = inf.infExact()
    # if covFTIC then infFITC
    

    post  = analyze(inffunc, meanfunc, covfunc, likfunc, x, y, der=False)[1]
    alpha = post.alpha
    L     = post.L
    sW    = post.sW

    #if issparse(alpha)                  # handle things for sparse representations
    #    nz = alpha != 0                 # determine nonzero indices
    #    if issparse(L), L = full(L(nz,nz)); end      # convert L and sW if necessary
    #    if issparse(sW), sW = full(sW(nz)); end
    #else:

    nz = range(len(alpha[:,0]))      # non-sparse representation 
    if L == []:                      # in case L is not provided, we compute it
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
    if ys == None:
        return [ymu, ys2, fmu, fs2, None]
    else:
        return [ymu, ys2, fmu, fs2, lp]   



"""


