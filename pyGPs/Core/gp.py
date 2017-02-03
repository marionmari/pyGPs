from __future__ import division
from __future__ import absolute_import
from builtins import str
from builtins import range
from builtins import object
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
#   nlZ          returned value of the negative log marginal likelihood
#   dnlZ         column vector of partial derivatives of the negative
#                    log marginal likelihood w.r.t. each hyperparameter
#   ym           column vector (of length ns) of predictive output means
#   ys2          column vector (of length ns) of predictive output variances
#   fm           column vector (of length ns) of predictive latent means
#   fs2          column vector (of length ns) of predictive latent variances
#   lp           column vector (of length ns) of log predictive probabilities
#   post         struct representation of the (approximate) posterior
#                post consists of post.alpha, post.L, post.sW
#
# This is a object-oriented python implementation of gpml functionality
# (Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18).
# based on the functional-version of python implementation
# (Copyright (c) by Marion Neumann and Daniel Marthaler, 20/05/2013)
#
# Copyright (c) by Marion Neumann and Shan Huang, 30/09/2013

import itertools
import numpy as np
import matplotlib.pyplot as plt
from . import inf, mean, lik, cov, opt
from .tools import unique, jitchol, solve_chol
from copy import deepcopy
import pyGPs
from pyGPs.Core.cov import FITCOfKernel

SHADEDCOLOR = [0.7539, 0.89453125, 0.62890625, 1.0]
MEANCOLOR = [ 0.2109375, 0.63385, 0.1796875, 1.0]
DATACOLOR = [0.12109375, 0.46875, 1., 1.0]

class GP(object):
    '''
    Base class for GP model.
    '''
    def __init__(self):
        super(GP, self).__init__()
        self.usingDefaultMean = True  # was using default mean function now?
        self.meanfunc = None      # mean function
        self.covfunc = None       # covariance function
        self.likfunc = None       # likelihood function
        self.inffunc = None       # inference function
        self.optimizer = None     # optimizer object
        self.nlZ = None           # negative log marginal likelihood
        self.dnlZ = None          # column vector of partial derivatives of the negative
                                  # log marginal likelihood w.r.t. each hyperparameter
        self.posterior = None     # struct representation of the (approximate) posterior
        self.x = None             # n by D matrix of training inputs
        self.y = None             # column vector of length n of training targets
        self.xs = None            # n by D matrix of test inputs
        self.ys = None            # column vector of length nn of true test targets (optional)
        self.ym = None            # column vector (of length ns) of predictive output means
        self.ys2 = None           # column vector (of length ns) of predictive output variances
        self.fm = None            # column vector (of length ns) of predictive latent means
        self.fs2 = None           # column vector (of length ns) of predictive latent variances
        self.lp = None            # column vector (of length ns) of log predictive probabilities



    def __str__(self):
       strvalue = 'To get the properties of the model use:\n'+\
                  'model.nlZ          # negative log marginal likelihood\n'+\
                  'model.dnlZ.cov     # derivatives of cov func of negative log marginal likelihood\n'+\
                  'model.dnlZ.lik     # derivatives of lik func of negative log marginal likelihood\n'+\
                  'model.dnlZ.mean    # derivatives of mean func of negative log marginal likelihood\n'+\
                  'model.posterior    # posterior structure\n'+\
                  'model.covfunc.hyp  # hyperparameters of cov func\n'+\
                  'model.meanfunc.hyp # hyperparameters of mean func\n'+\
                  'model.likfunc.hyp  # hyperparameters of lik func\n'+\
                  'model.fm           # latent mean\n'+\
                  'model.fs2          # latent variance\n'+\
                  'model.ym           # predictive mean\n'+\
                  'model.ys2          # predictive variance\n'+\
                  'model.lp           # log predictive probability'
       return strvalue



    def __repr__(self):
       strvalue = str(type(self))+': '+\
                  'to get the properties of the model use:\n'+\
                  'model.nlZ          # negative log marginal likelihood\n'+\
                  'model.dnlZ.cov     # derivatives of cov func of negative log marginal likelihood\n'+\
                  'model.dnlZ.lik     # derivatives of lik func of negative log marginal likelihood\n'+\
                  'model.dnlZ.mean    # derivatives of mean func of negative log marginal likelihood\n'+\
                  'model.posterior    # posterior structure\n'+\
                  'model.covfunc.hyp  # hyperparameters of cov func\n'+\
                  'model.meanfunc.hyp # hyperparameters of mean func\n'+\
                  'model.likfunc.hyp  # hyperparameters of lik func\n'+\
                  'model.fm           # latent mean\n'+\
                  'model.fs2          # latent variance\n'+\
                  'model.ym           # predictive mean\n'+\
                  'model.ys2          # predictive variance\n'+\
                  'model.lp           # log predictive probability'
       return strvalue




    def setData(self, x, y):
        '''
        Set training inputs and traning labels to model.

        :param x: training inputs in shape (n,D)
        :param y: training labels in shape (n,1)

        Note this method will transform x, y to correct shape
        if x, y is given in 1d array. 
        '''
        # check wether the number of inputs and labels match
        assert x.shape[0] == y.shape[0], "number of inputs and labels does not match"

        # check the shape of inputs
        # transform to the correct shape
        if x.ndim == 1:
            x = np.reshape(x, (x.shape[0],1))
        if y.ndim == 1:
            y = np.reshape(y, (y.shape[0],1))

        self.x = x
        self.y = y
        if self.usingDefaultMean:
            c = np.mean(y)
            self.meanfunc = mean.Const(c)    # adapt default prior mean wrt. training labels



    def plotData_1d(self, axisvals=None):
        '''
        Toy Method for ploting 1d data of the model.

        :param list axisvals: [min_x, max_x, min_y, max_y] setting the plot range
        '''
        plt.figure()
        plt.plot(self.x, self.y, ls='None', marker='+', color=DATACOLOR, ms=12, mew=2)
        if axisvals:
            plt.axis(axisvals)
        plt.grid()
        plt.xlabel('input x')
        plt.ylabel('target y')
        plt.show()



    def plotData_2d(self,x1,x2,t1,t2,p1,p2,axisvals=None):
        '''
        Toy Method for ploting 2d data of the model. \n
        For plotting, we superimpose the data points with the posterior equi-probability contour 
        lines for the probability of class two given complete information about the generating mechanism.

        :param x1: inputs for class +1
        :param x2: inputs for class -1
        :param t1: meshgrid array for the first axis
        :param t2: meshgrid array for the second axis
        :param p1,p2: contour lines contains p2/(p1+p2)
        :param list axisvals: [min_x, max_x, min_y, max_y] setting the plot range

        That is to say, the contour is ploted by plt.contour(t1, t2, p2/(p1+p2) )
        Note these parameters are (only) used for our hard-coded data for classification demo.
        '''
        fig = plt.figure()
        plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
        plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
        pc = plt.contour(t1, t2, np.reshape(old_div(p2,(p1+p2)), (t1.shape[0],t1.shape[1]) ))
        fig.colorbar(pc)
        plt.grid()
        if axisvals:
            plt.axis(axisvals)
        plt.show()



    def setPrior(self, mean=None, kernel=None):
        '''
        Set prior mean and covariance other than the default setting of current model.

        :param mean: instance of mean class. (e.g. mean.Linear())
        :param kernel: instance of covariance class. (e.g. cov.RBF())
        '''
        # check the type of inputs
        # ensure they are the right class before setting prior
        if not mean is None:
            assert isinstance(mean, pyGPs.mean.Mean), "mean function is not an instance of pyGPs.mean.Mean"
            self.meanfunc = mean
            self.usingDefaultMean = False
        if not kernel is None:
            assert isinstance(kernel, pyGPs.cov.Kernel), "cov function is not an instance of pyGPs.cov.Kernel"
            self.covfunc = kernel
            if type(kernel) is cov.Pre:
                self.usingDefaultMean = False



    def setOptimizer(self, method, num_restarts=None, min_threshold=None, meanRange=None, covRange=None, likRange=None):
        '''
        This method is used to sepecify optimization configuration. By default, gp uses a single run "minimize".

        :param method: Optimization methods. Possible values are:\n
                       "Minimize"   -> minimize by Carl Rasmussen (python implementation of "minimize" in GPML)\n
                       "CG"         -> conjugent gradient\n
                       "BFGS"       -> quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)\n
                       "SCG"        -> scaled conjugent gradient (faster than CG)\n
        :param num_restarts: Set if you want to run mulitiple times of optimization with different initial guess.
                             It specifys the maximum number of runs/restarts/trials.
        :param min_threshold: Set if you want to run mulitiple times of optimization with different initial guess.
                              It specifys the threshold of objective function value. Stop optimization when this value is reached.
        :param meanRange: The range of initial guess for mean hyperparameters.
                          e.g. meanRange = [(-2,2), (-5,5), (0,1)].
                          Each tuple specifys the range (low, high) of this hyperparameter,
                          This is only the range of initial guess, during optimization process, optimal hyperparameters may go out of this range.
                          (-5,5) for each hyperparameter by default.
        :param covRange: The range of initial guess for kernel hyperparameters. Usage see meanRange
        :param likRange: The range of initial guess for likelihood hyperparameters. Usage see meanRange
        '''
        pass



    def optimize(self, x=None, y=None, numIterations=40):
        '''
        Train optimal hyperparameters based on training data,
        adjust new hyperparameters to all mean/cov/lik functions.

        :param x: training inputs in shape (n,D)
        :param y: training labels in shape (n,1)
        '''
        # check wether the number of inputs and labels match
        if x is not None and y is not None: 
            assert x.shape[0] == y.shape[0], "number of inputs and labels does not match"

        # check the shape of inputs
        # transform to the correct shape
        if not x is None:
            if x.ndim == 1:
                x = np.reshape(x, (x.shape[0],1))
            self.x = x

        if not y is None:
            if y.ndim == 1:
                y = np.reshape(y, (y.shape[0],1))
            self.y = y

        if self.usingDefaultMean and self.meanfunc is None:
            c = np.mean(y)
            self.meanfunc = mean.Const(c)    # adapt default prior mean wrt. training labels

        # optimize
        optimalHyp, optimalNlZ = self.optimizer.findMin(self.x, self.y, numIters = numIterations)
        self.nlZ = optimalNlZ

        # apply optimal hyp to all mean/cov/lik functions here
        self.optimizer._apply_in_objects(optimalHyp)
        self.getPosterior()



    def getPosterior(self, x=None, y=None, der=True):
        '''
        Fit the training data. Update negative log marginal likelihood(nlZ),
        partial derivatives of nlZ w.r.t. each hyperparameter(dnlZ),
        and struct representation of the (approximate) posterior(post),
        which consists of post.alpha, post.L, post.sW.

        nlZ, dnlZ, post = getPosterior(x, y, der=True)\n
        nlZ, post       = getPosterior(x, y, der=False )

        :param x: training inputs in shape (n,D)
        :param y: training labels in shape (n,1)
        :param boolean der: flag for whether to compute derivatives

        :return: negative log marginal likelihood (nlZ), derivatives of nlZ (dnlZ), posterior structure(post)

        You can print post to see descriptions of posterior.
        or see pyGPs.Core.inf for details.
        '''

        # check wether the number of inputs and labels match
        if x is not None and y is not None: 
            assert x.shape[0] == y.shape[0], "number of inputs and labels does not match"

        # check the shape of inputs
        # transform to the correct shape
        if not x is None:
            if x.ndim == 1:
                x = np.reshape(x, (x.shape[0],1))
            self.x = x

        if not y is None:
            if y.ndim == 1:
                y = np.reshape(y, (y.shape[0],1))
            self.y = y

        if self.usingDefaultMean and self.meanfunc is None:
            c = np.mean(y)
            self.meanfunc = mean.Const(c)    # adapt default prior mean wrt. training labels
            
        # call inference method
        if isinstance(self.likfunc, lik.Erf): #or is instance(self.likfunc, lik.Logistic):
            uy  = unique(self.y)
            ind = ( uy != 1 )
            if any( uy[ind] != -1):
                raise Exception('You attempt classification using labels different from {+1,-1}')
        if not der:
            post, nlZ = self.inffunc.evaluate(self.meanfunc, self.covfunc, self.likfunc, self.x, self.y, 2)
            self.nlZ = nlZ
            self.posterior = deepcopy(post)
            return nlZ, post
        else:
            post, nlZ, dnlZ = self.inffunc.evaluate(self.meanfunc, self.covfunc, self.likfunc, self.x, self.y, 3)
            self.nlZ       = nlZ
            self.dnlZ      = deepcopy(dnlZ)
            self.posterior = deepcopy(post)
            return nlZ, dnlZ, post



    def predict(self, xs, ys=None):
        '''
        Prediction of test points (given by xs) based on training data of the current model.
        This method will output the following value:\n
        predictive output means(ym),\n
        predictive output variances(ys2),\n
        predictive latent means(fm),\n
        predictive latent variances(fs2),\n
        log predictive probabilities(lp).\n
        Theses values can also be achieved from model's property. (e.g. model.ym)

        :param xs: test input in shape of nn by D
        :param ys: test target(optional) in shape of nn by 1 if given

        :return: ym, ys2, fm, fs2, lp
        '''
        # check the shape of inputs
        # transform to correct shape if neccessary
        if xs.ndim == 1:
            xs = np.reshape(xs, (xs.shape[0],1))
        self.xs = xs
        if not ys is None:
            if ys.ndim == 1:
                ys = np.reshape(ys, (ys.shape[0],1))
            self.ys = ys

        meanfunc = self.meanfunc
        covfunc  = self.covfunc
        likfunc  = self.likfunc
        inffunc  = self.inffunc
        x = self.x
        y = self.y

        if self.posterior is None:
            self.getPosterior()
        alpha = self.posterior.alpha
        L     = self.posterior.L
        sW    = self.posterior.sW

        nz = list(range(len(alpha[:,0])))         # non-sparse representation
        if len(L) == 0:                         # in case L is not provided, we compute it
            K = covfunc.getCovMatrix(x=x[nz,:], mode='train')
            #L = np.linalg.cholesky( (np.eye(nz) + np.dot(sW,sW.T)*K).T )
            L = jitchol( (np.eye(len(nz)) + np.dot(sW,sW.T)*K).T )
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
            id  = list(range(nact,min(nact+nperbatch,ns)))   # data points to process
            kss = covfunc.getCovMatrix(z=xs[id,:], mode='self_test')    # self-variances
            if isinstance(covfunc, FITCOfKernel): 
                Ks = covfunc.getCovMatrix(x=x, z=xs[id,:], mode='cross')   # cross-covariances
                Ks = Ks[nz,:]
            else:
                Ks  = covfunc.getCovMatrix(x=x[nz,:], z=xs[id,:], mode='cross')   # cross-covariances
            ms  = meanfunc.getMean(xs[id,:])
            N   = (alpha.shape)[1]                     # number of alphas (usually 1; more in case of sampling)
            Fmu = np.tile(ms,(1,N)) + np.dot(Ks.T,alpha[nz])          # conditional mean fs|f
            fmu[id] = np.reshape(old_div(Fmu.sum(axis=1),N),(len(id),1))       # predictive means
            if Ltril: # L is triangular => use Cholesky parameters (alpha,sW,L)
                V       = np.linalg.solve(L.T,np.tile(sW,(1,len(id)))*Ks)
                fs2[id] = kss - np.array([(V*V).sum(axis=0)]).T             # predictive variances
            else:     # L is not triangular => use alternative parametrization
                fs2[id] = kss + np.array([(Ks*np.dot(L,Ks)).sum(axis=0)]).T # predictive variances
            fs2[id] = np.maximum(fs2[id],0)            # remove numerical noise i.e. negative variances
            Fs2 = np.tile(fs2[id],(1,N))               # we have multiple values in case of sampling
            if ys is None:
                Lp, Ymu, Ys2 = likfunc.evaluate(None,Fmu[:],Fs2[:],None,None,3)
            else:
                Lp, Ymu, Ys2 = likfunc.evaluate(np.tile(ys[id],(1,N)), Fmu[:], Fs2[:],None,None,3)
            lp[id]  = np.reshape( old_div(np.reshape(Lp,(np.prod(Lp.shape),N)).sum(axis=1),N) , (len(id),1) )   # log probability; sample averaging
            ymu[id] = np.reshape( old_div(np.reshape(Ymu,(np.prod(Ymu.shape),N)).sum(axis=1),N) ,(len(id),1) )  # predictive mean ys|y and ...
            ys2[id] = np.reshape( old_div(np.reshape(Ys2,(np.prod(Ys2.shape),N)).sum(axis=1),N) , (len(id),1) ) # .. variance
            nact = id[-1]+1                  # set counter to index of next data point
        self.ym = ymu
        self.ys2 = ys2
        self.lp = lp
        self.fm = fmu
        self.fs2 = fs2
        if ys is None:
            return ymu, ys2, fmu, fs2, None
        else:
            return ymu, ys2, fmu, fs2, lp



    def predict_with_posterior(self, post, xs, ys=None):
        '''
        Prediction of test points (given by xs) based on training data 
        of the current model with posterior already provided.
        (i.e. you already have the posterior and thus don't need the fitting phase.)
        This method will output the following value:\n
        predictive output means(ym),\n
        predictive output variances(ys2),\n
        predictive latent means(fm),\n
        predictive latent variances(fs2),\n
        log predictive probabilities(lp).\n
        Theses values can also be achieved from model's property. (e.g. model.ym)

        :param post: struct representation of posterior
        :param xs: test input
        :param ys: test target(optional)

        :return: ym, ys2, fm, fs2, lp
        '''
        # check the shape of inputs
        # transform to correct shape if neccessary
        if xs.ndim == 1:
            xs = np.reshape(xs, (xs.shape[0],1))
        self.xs = xs
        if not ys is None:
            if ys.ndim == 1:
                ys = np.reshape(ys, (ys.shape[0],1))
            self.ys = ys

        meanfunc = self.meanfunc
        covfunc  = self.covfunc
        likfunc  = self.likfunc
        inffunc  = self.inffunc
        x = self.x
        y = self.y

        self.posterior = deepcopy(post)
        alpha = post.alpha
        L     = post.L
        sW    = post.sW

        nz = list(range(len(alpha[:,0])))         # non-sparse representation
        if len(L) == 0:                         # in case L is not provided, we compute it
            K = covfunc.getCovMatrix(x=x[nz,:], mode='train')
            #L = np.linalg.cholesky( (np.eye(nz) + np.dot(sW,sW.T)*K).T )
            L = jitchol( (np.eye(len(nz)) + np.dot(sW,sW.T)*K).T )
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
            id  = list(range(nact,min(nact+nperbatch,ns)))   # data points to process
            kss = covfunc.getCovMatrix(z=xs[id,:], mode='self_test')    # self-variances
            Ks  = covfunc.getCovMatrix(x=x[nz,:], z=xs[id,:], mode='cross')   # cross-covariances
            ms  = meanfunc.getMean(xs[id,:])
            N   = (alpha.shape)[1]                     # number of alphas (usually 1; more in case of sampling)
            Fmu = np.tile(ms,(1,N)) + np.dot(Ks.T,alpha[nz])          # conditional mean fs|f
            fmu[id] = np.reshape(old_div(Fmu.sum(axis=1),N),(len(id),1))       # predictive means
            if Ltril: # L is triangular => use Cholesky parameters (alpha,sW,L)
                V       = np.linalg.solve(L.T,np.tile(sW,(1,len(id)))*Ks)
                fs2[id] = kss - np.array([(V*V).sum(axis=0)]).T             # predictive variances
            else:     # L is not triangular => use alternative parametrization
                fs2[id] = kss + np.array([(Ks*np.dot(L,Ks)).sum(axis=0)]).T # predictive variances
            fs2[id] = np.maximum(fs2[id],0)            # remove numerical noise i.e. negative variances
            Fs2 = np.tile(fs2[id],(1,N))               # we have multiple values in case of sampling
            if ys is None:
                [Lp, Ymu, Ys2] = likfunc.evaluate(None,Fmu[:],Fs2[:],None,None,3)
            else:
                [Lp, Ymu, Ys2] = likfunc.evaluate(np.tile(ys[id],(1,N)), Fmu[:], Fs2[:],None,None,3)
            lp[id]  = np.reshape( old_div(np.reshape(Lp,(np.prod(Lp.shape),N)).sum(axis=1),N) , (len(id),1) )   # log probability; sample averaging
            ymu[id] = np.reshape( old_div(np.reshape(Ymu,(np.prod(Ymu.shape),N)).sum(axis=1),N) ,(len(id),1) )  # predictive mean ys|y and ...
            ys2[id] = np.reshape( old_div(np.reshape(Ys2,(np.prod(Ys2.shape),N)).sum(axis=1),N) , (len(id),1) ) # .. variance
            nact = id[-1]+1                  # set counter to index of next data point
        self.ym  = ymu
        self.ys2 = ys2
        self.lp  = lp
        self.fm  = fmu
        self.fs2 = fs2
        if ys is None:
            return ymu, ys2, fmu, fs2, None
        else:
            return ymu, ys2, fmu, fs2, lp





class GPR(GP):
    '''
    Model for Gaussian Process Regression
    '''
    def __init__(self):
        super(GPR, self).__init__()
        self.meanfunc = mean.Zero()                        # default prior mean
        self.covfunc = cov.RBF()                           # default prior covariance
        self.likfunc = lik.Gauss()                         # likihood with default noise variance 0.1
        self.inffunc = inf.Exact()                         # inference method
        self.optimizer = opt.Minimize(self)                # default optimizer



    def setNoise(self,log_sigma):
        '''
        Set noise other than default noise value

        :param log_sigma: logarithm of the noise sigma
        '''
        self.likfunc = lik.Gauss(log_sigma)


    def setOptimizer(self, method, num_restarts=None, min_threshold=None, meanRange=None, covRange=None, likRange=None):
        '''
        Overriding. Usage see base class pyGPs.gp.GP.setOptimizer
        '''
        conf = None
        if (num_restarts!=None) or (min_threshold!=None):
            conf = pyGPs.Optimization.conf.random_init_conf(self.meanfunc,self.covfunc,self.likfunc)
            conf.num_restarts = num_restarts
            conf.min_threshold = min_threshold
            if not meanRange is None:
                conf.meanRange = meanRange
            if not covRange is None:
                conf.covRange = covRange
            if not likRange is None:
                conf.likRange = likRange
        if method == "Minimize":
            self.optimizer = opt.Minimize(self,conf)
        elif method == "SCG":
            self.optimizer = opt.SCG(self,conf)
        elif method == "CG":
            self.optimizer = opt.CG(self,conf)
        elif method == "BFGS":
            self.optimizer = opt.BFGS(self,conf)
        elif method == "LBFGSB":
            self.optimizer = opt.LBFGSB(self, conf)
        elif method == "COBYLA":
            self.optimizer = opt.COBYLA(self, conf)
        elif method == "RTMinimize":
            self.optimizer = opt.RTMinimize(self, conf)
        else:
            raise Exception('Optimization method is not set correctly in setOptimizer')


    def plot(self,axisvals=None):
        '''
        Plot 1d GP regression result.

        :param list axisvals: [min_x, max_x, min_y, max_y] setting the plot range
        '''
        xs = self.xs    # test point
        x = self.x
        y = self.y
        ym = self.ym    # predictive test mean
        ys2 = self.ys2  # predictive test variance
        plt.figure()
        xss  = np.reshape(xs,(xs.shape[0],))
        ymm  = np.reshape(ym,(ym.shape[0],))
        ys22 = np.reshape(ys2,(ys2.shape[0],))
        plt.plot(x, y, color=DATACOLOR, ls='None', marker='+',ms=12, mew=2)
        plt.plot(xs, ym, color=MEANCOLOR, ls='-', lw=3.)
        plt.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=SHADEDCOLOR,linewidths=0.0)
        plt.grid()
        if not axisvals is None:
            plt.axis(axisvals)
        plt.xlabel('input x')
        plt.ylabel('target y')
        plt.show()


    def useInference(self, newInf):
        '''
        Use another inference techinique other than default exact inference.

        :param str newInf: 'Laplace' or 'EP'
        '''
        if newInf == "Laplace":
            self.inffunc = inf.Laplace()
        elif newInf == "EP":
            self.inffunc = inf.EP()
        else:
            raise Exception('Possible inf values are "Laplace", "EP".')


    def useLikelihood(self,newLik):
        '''
        Use another likelihood function other than default Gaussian likelihood.

        :param str newLik: 'Laplace'
        '''
        if newLik == "Laplace":
            self.likfunc = lik.Laplace()
            self.inffunc = inf.EP()
        else:
            raise Exception('Possible lik values are "Laplace".')





class GPC(GP):
    '''
    Model for Gaussian Process Classification.
    '''
    def __init__(self):
        super(GPC, self).__init__()
        self.meanfunc = mean.Zero()                        # default prior mean
        self.covfunc = cov.RBF()                           # default prior covariance
        self.likfunc = lik.Erf()                           # erf likihood
        self.inffunc = inf.EP()                            # default inference method
        self.optimizer = opt.Minimize(self)                # default optimizer



    def setOptimizer(self, method, num_restarts=None, min_threshold=None, meanRange=None, covRange=None, likRange=None):
        '''
        Overriding. Usage see base class pyGPs.gp.GP.setOptimizer
        '''
        conf = None
        if (num_restarts!=None) or (min_threshold!=None):
            conf = pyGPs.Optimization.conf.random_init_conf(self.meanfunc,self.covfunc,self.likfunc)
            conf.num_restarts = num_restarts
            conf.min_threshold = min_threshold
            if not meanRange is None:
                conf.meanRange = meanRange
            if not covRange is None:
                conf.covRange = covRange
            if not likRange is None:
                conf.likRange = likRange
        if method == "Minimize":
            self.optimizer = opt.Minimize(self,conf)
        elif method == "SCG":
            self.optimizer = opt.SCG(self,conf)
        elif method == "CG":
            self.optimizer = opt.CG(self,conf)
        elif method == "BFGS":
            self.optimizer = opt.BFGS(self,conf)



    def plot(self,x1,x2,t1,t2,axisvals=None):
        '''
        Plot 2d GP Classification result.

        For plotting, we superimpose the data points with the posterior equi-probability contour 
        lines for the probability of class two given complete information about the generating mechanism.

        :param x1: inputs for class +1
        :param x2: inputs for class -1
        :param t1: meshgrid array for the first axis
        :param t2: meshgrid array for the second axis
        :param list axisvals: [min_x, max_x, min_y, max_y] setting the plot range

        Note these parameters are (only) used for our hard-coded data for classification demo.
        '''
        fig = plt.figure()
        plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
        plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
        pc = plt.contour(t1, t2, np.reshape(np.exp(self.lp), (t1.shape[0],t1.shape[1]) ))
        fig.colorbar(pc)
        plt.grid()
        if not axisvals is None:
            plt.axis(axisvals)
        plt.show()



    def useInference(self, newInf):
        '''
        Use another inference techinique other than default EP inference.

        :param str newInf: 'Laplace'
        '''
        if newInf == "Laplace":
            self.inffunc = inf.Laplace()
        else:
            raise Exception('Possible inf values are "Laplace".')



    def useLikelihood(self,newLik):
        '''
        Use another likelihood function other than default error function.
        (Not used in this version)

        :param str newLik: 'Logistic'
        '''
        if newLik == "Logistic":
            raise Exception("Logistic likelihood is currently not implemented.")
            #self.likfunc = lik.Logistic()
        else:
            raise Exception('Possible lik values are "Logistic".')





class GPMC(object):
    '''
    This is a one vs. one classification wrapper for GP Classification
    '''
    def __init__(self, n_class):
        self.meanfunc = mean.Zero()                # default prior mean
        self.covfunc = cov.RBF()                   # default prior covariance
        self.n_class = n_class                     # number of different classes
        self.x_all = None
        self.y_all = None
        self.newInf = None                         # new inference? -> call useInference
        self.newLik = None                         # new likelihood? -> call useLikelihood
        self.newPrior = False



    def setPrior(self, mean=None, kernel=None):
        '''
        Set prior mean and covariance other than the default setting of current model.

        :param mean: instance of mean class. (e.g. mean.Linear())
        :param kernel: instance of covariance class. (e.g. cov.RBF())
        '''
        # check the type of inputs
        # ensure they are the right class before setting prior
        if not mean is None:
            assert isinstance(mean, pyGPs.mean.Mean), "mean function is not an instance of pyGPs.mean.Mean"
            self.meanfunc = mean
            self.usingDefaultMean = False
        if not kernel is None:
            assert isinstance(kernel, pyGPs.cov.Kernel), "cov function is not an instance of pyGPs.cov.Kernel"
            self.covfunc = kernel
            if type(kernel) is cov.Pre:
                self.usingDefaultMean = False
        self.newPrior = True



    def useInference(self, newInf):
        '''
        Use another inference techinique other than default EP inference.

        :param str newInf: 'Laplace'
        '''
        if newInf == "Laplace":
            self.inffunc = inf.Laplace()
        else:
            raise Exception('Possible inf values are "Laplace".')



    def useLikelihood(self,newLik):
        '''
        Use another likelihood function other than default error function.
        (Not used in this version)

        :param str newLik: 'Logistic'
        '''
        if newLik == "Logistic":
            raise Exception("Logistic likelihood is currently not implemented.")
            #self.likfunc = lik.Logistic()
        else:
            raise Exception('Possible lik values are "Logistic".')



    def setData(self,x,y):
        '''
        Set training inputs and traning labels to model.

        :param x: training inputs in shape (n,D)
        :param y: training labels in shape (n,1)

        Note this method will transform x, y to correct shape
        if x, y is given in 1d array. 
        '''
        # check wether the number of inputs and labels match
        assert x.shape[0] == y.shape[0], "number of inputs and labels does not match"

        # check the shape of inputs
        # transform to the correct shape
        if x.ndim == 1:
            x = np.reshape(x, (x.shape[0],1))
        if y.ndim == 1:
            y = np.reshape(y, (y.shape[0],1))

        self.x_all = x
        self.y_all = y



    def fitAndPredict(self, xs):
        '''
        Fit the model with given training data and predict for test points (given by xs).
        predictive_vote is a matrix where row i is each test point i,
        and column j is the probability for being class j

        :param xs: test inputs in shape of nn by D
        :return: predictive_vote
        '''
        # check the shape of inputs
        if xs.ndim == 1:
            xs = np.reshape(xs, (xs.shape[0],1))

        predictive_vote = np.zeros((xs.shape[0],self.n_class))
        for i in range(self.n_class):         # classifier for class i...
            for j in range(i+1,self.n_class): # ...and class j
                x,y = self.createBinaryClass(i,j)
                model = GPC()
                if self.newPrior:
                    model.setPrior(mean=self.meanfunc, kernel=self.covfunc)
                if self.newInf:
                    model.useInference(self.newInf)
                if self.newLik:
                    model.useLikelihood(self.newLik)
                model.getPosterior(x,y)               # fitting
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



    def optimizeAndPredict(self, xs):
        '''
        Optimize the model with given training data and predict for test points (given by xs).
        predictive_vote is a matrix where row i is each test point i,
        and column j is the probability for being class j

        :param xs: test inputs in shape of nn by D
        :return: predictive_vote
        '''
        # check the shape of inputs
        if xs.ndim == 1:
            xs = np.reshape(xs, (xs.shape[0],1))

        predictive_vote = np.zeros((xs.shape[0],self.n_class))
        for i in range(self.n_class):         # classifier for class i...
            for j in range(i+1,self.n_class): # ...and class j
                x,y = self.createBinaryClass(i,j)
                model = GPC()
                if self.newPrior:
                    model.setPrior(mean=self.meanfunc, kernel=self.covfunc)
                if self.newInf:
                    model.useInference(self.newInf)
                if self.newLik:
                    model.useLikelihood(self.newLik)
                model.optimize(x,y)               # training
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



    def createBinaryClass(self, i,j):
        ''' 
        Create dataset x(data) and y(label) which only contains class i and j.
        Relabel class i to +1 and class j to -1

        :param int i: the i_th class
        :param int j: the j_th class
        :return: x(data) and y(label) which only contains class i and j

        '''
        class_i = []
        class_j = []
        for index in range(len(self.y_all)):       # check all classes
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
    '''
    Model for FITC GP base class
    '''
    def __init__(self):
        super(GP_FITC, self).__init__()
        self.u = None                  # inducing points



    def setData(self, x, y, value_per_axis=5):
        '''
        Set training inputs and traning labels to model and derive deault inducing_points..

        :param x: training inputs in shape (n,D)
        :param y: training labels in shape (n,1)
        :param int value_per_axis: number of value in each dimension
                                   when using a uni-distant default inducing points

        Note this method will transform x, y to correct shape
        if x, y is given in 1d array. 
        '''
        # check wether the number of inputs and labels match
        assert x.shape[0] == y.shape[0], "number of inputs and labels does not match"

        # check dimension of inputs
        # transform to correct shape if neccessary
        if x.ndim == 1:
            x = np.reshape(x, (x.shape[0],1))
        if y.ndim == 1:
            y = np.reshape(y, (y.shape[0],1))
        self.x = x
        self.y = y
        if self.usingDefaultMean:
            c = np.mean(y)
            self.meanfunc = mean.Const(c)    # adapt default prior mean wrt. training labels

        # get range of x in each dimension
        # 5 uniformally selected value for each dimension
        gridAxis=[]
        for d in range(x.shape[1]):
            column = x[:,d]
            mini = np.min(column)
            maxi = np.max(column)
            axis = np.linspace(mini,maxi,value_per_axis)
            gridAxis.append(axis)
        # default inducing points-> a grid
        if self.u is None:
            self.u = np.array(list(itertools.product(*gridAxis)))
            self.covfunc = self.covfunc.fitc(self.u)



    def setPrior(self, mean=None, kernel=None, inducing_points=None):
        '''
        Set prior mean and covariance other than the default setting of current model,
        as well as the inducing points

        :param mean: instance of mean class. (e.g. mean.Linear())
        :param kernel: instance of covariance class. (e.g. cov.RBF())
        :inducing_points: matrix of inducing points in shape of (nu,D)
        '''
        if not kernel is None:
            if not inducing_points is None:
                self.covfunc = kernel.fitc(inducing_points)
                self.u = inducing_points
            else:
                if not self.u is None:
                    self.covfunc = kernel.fitc(self.u)
                else:
                    raise Exception("To use default inducing points, please call setData() first!")
            if type(kernel) is cov.Pre:
                self.usingDefaultMean = False
        if not mean is None:
            self.meanfunc = mean
            self.usingDefaultMean = False





class GPR_FITC(GP_FITC):
    '''
    Model for Gaussian Process Regression FITC
    '''
    def __init__(self):
        super(GPR_FITC, self).__init__()
        self.meanfunc = mean.Zero()                        # default prior mean
        self.covfunc = cov.RBF()                           # default prior covariance
        self.likfunc = lik.Gauss()                         # likihood with default noise variance 0.1
        self.inffunc = inf.FITC_Exact()                    # inference method
        self.optimizer = opt.Minimize(self)                # default optimizer
        self.u = None                                      # no default inducing points



    def setNoise(self,log_sigma):
        '''
        Set noise other than default noise value

        :param log_sigma: logarithm of the noise sigma
        '''
        self.likfunc = lik.Gauss(log_sigma)



    def setOptimizer(self, method, num_restarts=None, min_threshold=None, meanRange=None, covRange=None, likRange=None):
        '''
        Overriding. Usage see base class pyGPs.gp.GP.setOptimizer
        '''
        conf = None
        if (num_restarts!=None) or (min_threshold!=None):
            conf = pyGPs.Optimization.conf.random_init_conf(self.meanfunc,self.covfunc,self.likfunc)
            conf.num_restarts = num_restarts
            conf.min_threshold = min_threshold
            if not meanRange is None:
                conf.meanRange = meanRange
            if not covRange is None:
                conf.covRange = covRange
            if not likRange is None:
                conf.likRange = likRange
        if method == "Minimize":
            self.optimizer = opt.Minimize(self,conf)
        elif method == "SCG":
            self.optimizer = opt.SCG(self,conf)
        elif method == "CG":
            self.optimizer = opt.CG(self,conf)
        elif method == "BFGS":
            self.optimizer = opt.BFGS(self,conf)



    def plot(self,axisvals=None):
        '''
        Plot 1d GP FITC Regression result.

        :param list axisvals: [min_x, max_x, min_y, max_y] setting the plot range
        '''
        plt.figure()
        xss  = np.reshape(self.xs,(self.xs.shape[0],))
        ymm  = np.reshape(self.ym,(self.ym.shape[0],))
        ys22 = np.reshape(self.ys2,(self.ys2.shape[0],))
        plt.plot(self.x, self.y, color=DATACOLOR, ls='None', marker='+',ms=12, mew=2)
        plt.plot(self.xs, self.ym, color=MEANCOLOR, ls='-', lw=3.)
        plt.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=SHADEDCOLOR,linewidths=0.0)
        plt.grid()
        if not axisvals is None:
            plt.axis(axisvals)
        plt.xlabel('input x')
        plt.ylabel('output y')
        plt.plot(self.u,np.ones_like(self.u), ls='None', color='k',marker='x',markersize=12,mew=2)
        plt.show()



    def useInference(self, newInf):
        '''
        Use another inference techinique other than default exact inference.

        :param str newInf: 'Laplace' or 'EP'
        '''
        if newInf == "Laplace":
            self.inffunc = inf.FITC_Laplace()
        elif newInf == "EP":
            self.inffunc = inf.FITC_EP()
        else:
            raise Exception('Possible inf values are "Laplace", "EP".')



    def useLikelihood(self,newLik):
        '''
        Use another inference techinique other than default Gaussian likelihood.

        :param str newLik: 'Laplace'
        '''
        if newLik == "Laplace":
            self.likfunc = lik.Laplace()
            self.inffunc = inf.FITC_EP()
        else:
            raise Exception('Possible lik values are "Laplace".')





class GPC_FITC(GP_FITC):
    '''
    Model for Gaussian Process Classification FITC
    '''
    def __init__(self):
        super(GPC_FITC, self).__init__()
        self.meanfunc = mean.Zero()                        # default prior mean
        self.covfunc = cov.RBF()                           # default prior covariance
        self.likfunc = lik.Erf()                           # erf liklihood
        self.inffunc = inf.FITC_EP()                       # default inference method
        self.optimizer = opt.Minimize(self)                # default optimizer
        self.u = None                                      # no default inducing points



    def setOptimizer(self, method, num_restarts=None, min_threshold=None, meanRange=None, covRange=None, likRange=None):
        '''
        Overriding. Usage see base class pyGPs.gp.GP.setOptimizer
        '''
        conf = None
        if (num_restarts!=None) or (min_threshold!=None):
            conf = pyGPs.Optimization.conf.random_init_conf(self.meanfunc,self.covfunc,self.likfunc)
            conf.num_restarts = num_restarts
            conf.min_threshold = min_threshold
            if not meanRange is None:
                conf.meanRange = meanRange
            if not covRange is None:
                conf.covRange = covRange
            if not likRange is None:
                conf.likRange = likRange
        if method == "Minimize":
            self.optimizer = opt.Minimize(self,conf)
        elif method == "SCG":
            self.optimizer = opt.SCG(self,conf)
        elif method == "CG":
            self.optimizer = opt.CG(self,conf)
        elif method == "BFGS":
            self.optimizer = opt.BFGS(self,conf)



    def plot(self,x1,x2,t1,t2,axisvals=None):
        '''Plot 2d GP FITC classification.
        For plotting, we superimpose the data points with the posterior equi-probability contour 
        lines for the probability of class two given complete information about the generating mechanism.

        :param x1: inputs for class +1
        :param x2: inputs for class -1
        :param t1: meshgrid array for the first axis
        :param t2: meshgrid array for the second axis
        :param list axisvals: [min_x, max_x, min_y, max_y] setting the plot range

        Note these parameters are (only) used for our hard-coded data for classification demo.
        '''
        fig = plt.figure()
        plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
        plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
        plt.plot(self.u[:,0],self.u[:,1],'ko', markersize=12)
        pc = plt.contour(t1, t2, np.reshape(np.exp(self.lp), (t1.shape[0],t1.shape[1]) ))
        fig.colorbar(pc)
        plt.grid()
        if not axisvals is None:
            plt.axis(axisvals)
        plt.show()



    def useInference(self, newInf):
        '''
        Use another inference techinique other than default exact inference.

        :param str newInf: 'Laplace' or 'EP'
        '''
        if newInf == "Laplace":
            self.inffunc = inf.FITC_Laplace()
        else:
            raise Exception('Possible inf values are "Laplace".')



    def useLikelihood(self,newLik):
        '''
        Use another inference techinique other than default Erf likelihood.
        (Not used in this version)

        :param str newLik: 'Logistic'
        '''
        if newLik == "Logistic":
            raise Exception("Logistic likelihood is currently not implemented.")
        else:
            raise Exception('Possible lik values are "Logistic".')




