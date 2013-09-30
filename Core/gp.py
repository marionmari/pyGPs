#===============================================================================
#    Copyright (C) 2013
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [marthaler at ge dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at iais dot fraunhofer dot de]
# 
#    Fraunhofer IAIS, STREAM Project, Sankt Augustin, Germany
# 
#    This file is part of pyGPs.
# 
#    pyGPs is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
# 
#    pyGPs is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, see <http://www.gnu.org/licenses/>.
#===============================================================================

import numpy as np
import inf
import mean
import lik
from tools import unique

#=======================================================================================
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
#======================================================================================
#
# @author: Shan Huang (last update Sep.2013)
# This is a object-oriented python implementation of gpml functionality 
# (Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18).
# based on the functional-version of python implementation
# (Copyright (c) by Marion Neumann and Daniel Marthaler, 20/05/2013)
# 
#
# Copyright (c) by Marion Neumann and Shan Huang, Sep.2013
#===============================================================================



def train(inffunc, meanfunc, covfunc, likfunc, x, y, optimizer):
    '''
    train optimal hyperparameters 
    adjust to all mean/cov/lik functions
    return nlZ after optimazation

    if you want to check hyp after training, it's now become property of each function
        (e.g.) print myCov.hyp
               print myMean.hyp
    '''
    if not meanfunc:
        meanfunc = mean.meanZero()
    if not covfunc:
        raise Exception('Covariance function cannot be empty')
    if not likfunc:
        likfunc = lik.likGauss([0.1])  
    if not inffunc:
        inffunc = inf.infExact()
    # if covFTIC then infFITC, leave this part aside now

    # optimize here
    out = optimizer.findMin(inffunc, meanfunc, covfunc, likfunc, x, y)
    optimalHyp = out[0] 
    optimalNlZ = out[1]

    # apply optimal hyp to all mean/cov/lik functions here
    optimizer.apply_in_objects(optimalHyp, meanfunc, covfunc, likfunc)
    return optimalNlZ
        


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



def analyze(inffunc, meanfunc, covfunc, likfunc, x, y, der=False):
    '''
    Middle Step, or maybe useful for experts to analyze sth.
    return [nlZ, post]        if der = False
        or [nlZ, dnlZ, post]  if der = True
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

    # call inference method
    if isinstance(likfunc, lik.likErf):  #or isinstance(likfunc, likelihood.likLogistic)
        uy = unique(y)        
        ind = ( uy != 1 )
        if any( uy[ind] != -1):
            raise Exception('You attempt classification using labels different from {+1,-1}')
    if not der:
        vargout = inffunc.proceed(meanfunc, covfunc, likfunc, x, y, 2) 
        post = vargout[0]
        nlZ = vargout[1] 
        return [nlZ, post]          # report -log marg lik, and post
    else:
        vargout = inffunc.proceed(meanfunc, covfunc, likfunc, x, y, 3) 
        post = vargout[0]
        nlZ = vargout[1]
        dnlZ = vargout[2] 
        return [nlZ, dnlZ, post]    # report -log marg lik, derivatives and post



