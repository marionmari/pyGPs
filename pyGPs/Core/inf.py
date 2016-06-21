from __future__ import division
from __future__ import absolute_import
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
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

# Inference methods: Compute the (approximate) posterior for a Gaussian process.
# Methods currently implemented include:
#
#   Exact         Exact inference (only possible with Gaussian likelihood)
#   Laplace       Laplace's Approximation
#   EP            Expectation Propagation
#   VB            [NOT IMPLEMENTED!] Variational Bayes Approximation
#
#   FITC          Large scale regression with approximate covariance matrix
#   FITC_Laplace  Large scale inference  with approximate covariance matrix
#   FITC_EP       Large scale inference  with approximate covariance matrix
#
#   MCMC     [NOT IMPLEMENTED!]
#               Markov Chain Monte Carlo and Annealed Importance Sampling
#               We offer two samplers.
#                 - hmc: Hybrid Monte Carlo
#                 - ess: Elliptical Slice Sampling
#               No derivatives w.r.t. to hyperparameters are provided.
#
#   LOO      [NOT IMPLEMENTED!]
#               Leave-One-Out predictive probability and Least-Squares Approxim.
#
#
# This is a object-oriented python implementation of gpml functionality
# (Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18).
# based on the functional-version of python implementation
# (Copyright (c) by Marion Neumann and Daniel Marthaler, 20/05/2013)
#
# Copyright (c) by Marion Neumann and Shan Huang, 30/09/2013


import numpy as np
from . import lik, cov
from copy import copy, deepcopy
from .tools import solve_chol, brentmin, cholupdate, jitchol
np.seterr(all='ignore')


class postStruct(object):
    '''
    Data structure for posterior

    | post.alpha: 1d array containing inv(K)*(mu-m), 
    |             where K is the prior covariance matrix, m the prior mean, 
    |             and mu the approx posterior mean
    | post.sW: 1d array containing diagonal of sqrt(W)
    |          the approximate posterior covariance matrix is inv(inv(K)+W)
    | post.L : 2d array, L = chol(sW*K*sW+identity(n))

    Usually, the approximate posterior to be returned admits the form
    N(mu=m+K*alpha, V=inv(inv(K)+W)), where alpha is a vector and W is diagonal;
    if not, then L contains instead -inv(K+inv(W)), and sW is unused.
    '''
    def __init__(self):
        self.alpha = np.array([])
        self.L     = np.array([])
        self.sW    = np.array([])

    def __repr__(self):
        value = "posterior: to get the parameters of the posterior distribution use:\n"+\
                "model.posterior.alpha\n"+"model.posterior.L\n"+"model.posterior.sW\n"+\
        "See documentation and gpml book chapter 2.3 and chapter 3.4.3 for these parameters."
        return value

    def __str__(self):
        value = "posterior distribution described by alpha, sW and L\n"+\
                "See documentation and gpml book chapter 2.3 and chapter 3.4.3 for these parameters\n"\
                +"alpha:\n"+str(self.alpha)+"\n"+"L:\n"+str(self.L)+"\nsW:\n"+str(self.sW)
        return value



class dnlZStruct(object):
    '''
    Data structure for the derivatives of mean, cov and lik functions.

    |dnlZ.mean: list of derivatives for each hyperparameters in mean function
    |dnlZ.cov: list of derivatives for each hyperparameters in covariance function
    |dnlZ.lik: list of derivatives for each hyperparameters in likelihood function
    '''
    def __init__(self, m, c, l):
        self.mean = []
        self.cov = []
        self.lik = []
        if m.hyp is not None:
            self.mean = [0 for i in range(len(m.hyp))]
        if c.hyp is not None:
            self.cov  = [0 for i in range(len(c.hyp))]
        if l.hyp is not None:
            self.lik  = [0 for i in range(len(l.hyp))]

    def __str__(self):
        value = "Derivatives of mean, cov and lik functions:\n" +\
                "mean:"+str(self.mean)+"\n"+\
                "cov:"+str(self.cov)+"\n"+\
                "lik:"+str(self.lik)
        return value

    def __repr__(self):
        value = "dnlZ: to get the derivatives of mean, cov and lik functions use:\n" +\
                "model.dnlZ.mean\n"+"model.dnlZ.cov\n"+"model.dnlZ.lik"
        return value

    def accumulateDnlZ(self,dnlZObject):
        self.mean= [x+y for x, y in zip(self.mean, dnlZObject.mean)]
        self.cov = [x+y for x, y in zip(self.cov, dnlZObject.cov)]
        self.lik = [x+y for x, y in zip(self.lik, dnlZObject.lik)]
        return self




class Inference(object):
    '''
    Base class for inference. Defined several tool methods in it.
    '''
    def __init__(self):
        pass

    def evaluate(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        '''
        Inference computation based on inputs.
        post, nlZ, dnlZ = inf.evaluate(mean, cov, lik, x, y)

            | INPUT:
            | cov: name of the covariance function (see covFunctions.m)
            | lik: name of the likelihood function (see likFunctions.m)
            | x: n by D matrix of training inputs
            | y: 1d array (of size n) of targets

            | OUTPUT:
            | post(postStruct): struct representation of the (approximate) posterior containing:
            | nlZ: returned value of the negative log marginal likelihood
            | dnlZ(dnlZStruct): struct representation for derivatives of the negative log marginal likelihood
            | w.r.t. each hyperparameter.

        Usually, the approximate posterior to be returned admits the form:
        N(m=K*alpha, V=inv(inv(K)+W)), where alpha is a vector and W is diagonal;
        if not, then L contains instead -inv(K+inv(W)), and sW is unused.

        For more information on the individual approximation methods and their
        implementations, see the respective inference function below. See also gp.py

        :param meanfunc: mean function
        :param covfunc: covariance function
        :param likfunc: likelihood function
        :param x: training data
        :param y: training labels
        :param nargout: specify the number of output(1,2 or 3)
        :return: posterior, negative-log-marginal-likelihood, derivative for negative-log-marginal-likelihood-likelihood
        '''
        pass

    def _epComputeParams(self, K, y, ttau, tnu, likfunc, m, inffunc):
        n     = len(y)                                                # number of training cases
        ssi   = np.sqrt(ttau)                                         # compute Sigma and mu
        #L     = np.linalg.cholesky(np.eye(n)+np.dot(ssi,ssi.T)*K).T   # L'*L=B=eye(n)+sW*K*sW
        L     = jitchol(np.eye(n)+np.dot(ssi,ssi.T)*K).T   # L'*L=B=eye(n)+sW*K*sW
        V     = np.linalg.solve(L.T,np.tile(ssi,(1,n))*K)
        Sigma = K - np.dot(V.T,V)
        mu    = np.dot(Sigma,tnu)
        Dsigma = np.reshape(np.diag(Sigma),(np.diag(Sigma).shape[0],1))
        tau_n = old_div(1,Dsigma) - ttau               # compute the log marginal likelihood
        nu_n  = old_div(mu,Dsigma)-tnu + m*tau_n       # vectors of cavity parameters
        lZ    = likfunc.evaluate(y, old_div(nu_n,tau_n), old_div(1,tau_n), inffunc)
        nlZ   = np.log(np.diag(L)).sum() - lZ.sum() - old_div(np.dot(tnu.T,np.dot(Sigma,tnu)),2)  \
                - old_div(np.dot((nu_n-m*tau_n).T,(old_div((ttau/tau_n*(nu_n-m*tau_n)-2*tnu), (ttau+tau_n)))),2) \
                + old_div((old_div(tnu**2,(tau_n+ttau))).sum(),2.)- old_div(np.log(1.+old_div(ttau,tau_n)).sum(),2.)
        return Sigma, mu, nlZ[0], L

    def _logdetA(self,K,w,nargout):
        '''
        Compute the log determinant ldA and the inverse iA of a square nxn matrix
        A = eye(n) + K*diag(w) from its LU decomposition; for negative definite A, we
        return ldA = Inf. We also return mwiA = -diag(w)*inv(A).
        [ldA,iA,mwiA] = _logdetA(K,w)'''
        n = K.shape[0]
        assert(K.shape[0] == K.shape[1])
        A = np.eye(n) + K*np.tile(w.T,(n,1))
        [L,U,P] = np.linalg.lu(A)    # compute LU decomposition, A = P'*L*U
        u = np.diag(U)
        signU = np.prod(np.sign(u))  # sign of U
        detP = 1                     # compute sign (and det) of the permutation matrix P
        p = np.dot(P,np.array(list(range(n))).T)
        for ii in range(n):
            if ii != p[ii]:
                detP = -detP
                j= [jj for jj,val in enumerate(p) if val == ii]
                p[ii,j[0]] = p[j[0],ii]
        if signU != detP:            # log becomes complex for negative values, encoded by infinity
            ldA = np.inf
        else:                        # det(L) = 1 and U triangular => det(A) = det(P)*prod(diag(U))
            ldA = np.log(np.abs(u)).sum()
        if nargout>1:
            iA = inv(U)*inv(L)*P;
            if nargout>2:
                mwiA = -np.tile(w,(1,n))*iA
                return ldA,iA,mwiA
            else:
                return ldA,iA
        else:
            return ldA

    def _Psi_line(self,s,dalpha,alpha,K,m,likfunc,y,inffunc):
        '''Criterion Psi at alpha + s*dalpha for line search
        [Psi,alpha,f,dlp,W] = _Psi_line(s,dalpha,alpha,hyp,K,m,lik,y,inf)
        '''
        alpha = alpha + s*dalpha
        f = np.dot(K,alpha) + m
        [lp,dlp,d2lp] = likfunc.evaluate(y,f,None,inffunc,None,3)
        W = -d2lp
        Psi = old_div(np.dot(alpha.T,(f-m)),2.) - lp.sum()
        return Psi[0],alpha,f,dlp,W

    def _epfitcZ(self,d,P,R,nn,gg,ttau,tnu,d0,R0,P0,y,likfunc,m,inffunc):
        '''
        Compute the marginal likelihood approximation
        effort is O(n*nu^2) provided that nu<n
        '''
        T = np.dot(np.dot(R,R0),P)              # temporary variable
        diag_sigma = d + np.array([(T*T).sum(axis=0)]).T
        mu = nn + np.dot(P.T,gg)                # post moments O(n*nu^2)
        tau_n = old_div(1.,diag_sigma)-ttau              # compute the log marginal likelihood
        nu_n  = old_div(mu,diag_sigma) - tnu + m*tau_n   # vectors of cavity parameters
        lZ = likfunc.evaluate(y, old_div(nu_n,tau_n), old_div(1,tau_n), inffunc, None, 1)
        nu = gg.shape[0]
        U = np.dot(R0,P0).T*np.tile(old_div(1,np.sqrt(d0+old_div(1,ttau))),(1,nu))
        #L = np.linalg.cholesky(np.eye(nu)+np.dot(U.T,U)).T
        L = jitchol(np.eye(nu)+np.dot(U.T,U)).T
        ld = 2.*np.log(np.diag(L)).sum() + (np.log(d0+old_div(1,ttau))).sum() + (np.log(ttau)).sum()
        t = np.dot(T,tnu); tnu_Sigma_tnu = np.dot(tnu.T,(d*tnu)) + np.dot(t.T,t)
        nlZ = old_div(ld,2.) - lZ.sum() -old_div(tnu_Sigma_tnu,2.) \
            -old_div(np.dot((nu_n-m*tau_n).T,(old_div((ttau/tau_n*(nu_n-m*tau_n)-2.*tnu),(ttau+tau_n)))),2.) \
            + old_div((old_div(tnu**2,(tau_n+ttau))).sum(),2.) - old_div((np.log(1+old_div(ttau,tau_n))).sum(),2.)
        return nlZ,nu_n,tau_n

    def _epfitcRefresh(self,d0,P0,R0,R0P0,w,b):
        '''
        Refresh the representation of the posterior from initial and site parameters
        to prevent possible loss of numerical precision after many epfitcUpdates
        effort is O(n*nu^2) provided that nu<n
        Sigma = inv(inv(K)+diag(W)) = diag(d) + P'*R0'*R'*R*R0*P.
        '''
        nu = R0.shape[0]                                 # number of inducing points
        rot180   = lambda A: np.rot90(np.rot90(A))       # little helper functions
        chol_inv = lambda A: rot180( np.linalg.solve(jitchol(rot180(A)), np.eye(nu)) ) # chol(inv(A))
    
        t  = old_div(1,(1+d0*w))                                  # temporary variable O(n)
        d  = d0*t                                        # O(n)
        P  = np.tile(t.T,(nu,1))*P0                      # O(n*nu)
        T  = np.tile((w*t).T,(nu,1))*R0P0                # temporary variable O(n*nu^2)
        R  = chol_inv(np.eye(nu)+np.dot(R0P0,T.T))       # O(n*nu^3)
        nn = d*b                                                  # O(n)
        gg = np.dot(R0.T,np.dot(R.T,np.dot(R,np.dot(R0P0,t*b))))  # O(n*nu)
        return d,P,R,nn,gg

    def _epfitcUpdate(self,d,P_i,R,nn,gg,w,b,ii,w_i,b_i,m,d0,P0,R0):
        dwi = w_i-w[ii]
        dbi = b_i-b[ii]
        hi = nn[ii] + m[ii] + np.dot(P_i.T,gg)           # posterior mean of site i O(nu)
        t = 1+dwi*d[ii]
        d[ii] = old_div(d[ii],t)                                  # O(1)
        nn[ii] = d[ii]*b_i                               # O(1)
        r = 1+d0[ii]*w[ii]
        r = old_div((r*r),dwi) + r*d0[ii]
        v = np.dot(R,np.dot(R0,P0[:,ii]))
        v = np.reshape(v,(v.shape[0],1))
        r = old_div(1,(r+np.dot(v.T,v)))
        if r>0:
            R = cholupdate(R,np.sqrt(r)*np.dot(R.T,v),'-')
        else:
            R = cholupdate(R,np.sqrt(-r)*np.dot(R.T,v),'+')
        ttemp = np.dot(R0.T,np.dot(R.T,np.dot(R,np.dot(R0,P_i))))
        gg = gg + (old_div((dbi-dwi*(hi-m[ii])),t)) * np.reshape(ttemp,(ttemp.shape[0],1)) # O(nu^2)
        w[ii] = w_i; b[ii] = b_i;                          # update site parameters O(1)
        P_i = old_div(P_i,t)                                        # O(nu)
        return d,P_i,R,nn,gg,w,b

    def _mvmZ(self,x,RVdd,t):
        '''
        Matrix vector multiplication with Z=inv(K+inv(W))
        '''
        Zx = t*x - np.dot(RVdd.T,np.dot(RVdd,x))
        return Zx

    def _mvmK(self,al,V,d0):
        '''
        Matrix vector multiplication with approximate covariance matrix
        '''
        Kal = np.dot(V.T,np.dot(V,al)) + d0*al
        return Kal

    def _Psi_lineFITC(self,s,dalpha,alpha,V,d0,m,likfunc,y,inffunc):
        '''
        Criterion Psi at alpha + s*dalpha for line search
        '''
        alpha = alpha + s*dalpha
        f = self._mvmK(alpha,V,d0) + m
        vargout = likfunc.evaluate(y,f,None,inffunc,None,3)
        lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]
        W = -d2lp
        Psi = old_div(np.dot(alpha.T,(f-m)),2.) - lp.sum()
        return Psi[0],alpha,f,dlp,W

    def _fitcRefresh(self,d0,P0,R0,R0P0, w):
        '''
        Refresh the representation of the posterior from initial and site parameters
        to prevent possible loss of numerical precision after many epfitcUpdates
        effort is O(n*nu^2) provided that nu<n
        Sigma = inv(inv(K)+diag(W)) = diag(d) + P'*R0'*R'*R*R0*P.
        '''
        nu = R0.shape[0]                                  # number of inducing points
        rot180   = lambda A: np.rot90(np.rot90(A))        # little helper functions
        #chol_inv = lambda A: np.linalg.solve( rot180( np.linalg.cholesky(rot180(A)) ),np.eye(nu)) # chol(inv(A))
        chol_inv = lambda A: rot180( np.linalg.solve(jitchol(rot180(A)), np.eye(nu)) ) # chol(inv(A))
 
        t  = old_div(1,(1+d0*w))                                   # temporary variable O(n)
        d  = d0*t                                         # O(n)
        P  = np.tile(t.T,(nu,1))*P0;                      # O(n*nu)
        T  = np.tile((w*t).T,(nu,1))*R0P0;                # temporary variable O(n*nu^2)
        R  = chol_inv(np.eye(nu)+np.dot(R0P0,T.T))        # O(n*nu^3)
        return d,P,R


class Exact(Inference):
    '''
    Exact inference for a GP with Gaussian likelihood. Compute a parametrization
    of the posterior, the negative log marginal likelihood and its derivatives
    w.r.t. the hyperparameters.
    '''
    def __init__(self):
        self.name = "Exact inference"
    def evaluate(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        if not isinstance(likfunc, lik.Gauss):
            raise Exception ('Exact inference only possible with Gaussian likelihood')
        n, D = x.shape
        K = covfunc.getCovMatrix(x=x, mode='train')            # evaluate covariance matrix
        m = meanfunc.getMean(x)                                # evaluate mean vector

        sn2   = np.exp(2*likfunc.hyp[0])                       # noise variance of likGauss
        #L     = np.linalg.cholesky(K/sn2+np.eye(n)).T         # Cholesky factor of covariance with noise
        L     = jitchol(old_div(K,sn2)+np.eye(n)).T            # Cholesky factor of covariance with noise
        alpha = old_div(solve_chol(L,y-m),sn2)
        post = postStruct()
        post.alpha = alpha                                     # return the posterior parameters
        post.sW    = old_div(np.ones((n,1)),np.sqrt(sn2))               # sqrt of noise precision vector
        post.L     = L                                         # L = chol(eye(n)+sW*sW'.*K)

        if nargout>1:                                          # do we want the marginal likelihood?
            nlZ = old_div(np.dot((y-m).T,alpha),2.) + np.log(np.diag(L)).sum() + n*np.log(2*np.pi*sn2)/2. # -log marg lik
            if nargout>2:                                      # do we want derivatives?
                dnlZ = dnlZStruct(meanfunc, covfunc, likfunc)  # allocate space for derivatives
                Q = old_div(solve_chol(L,np.eye(n)),sn2) - np.dot(alpha,alpha.T) # precompute for convenience
                dnlZ.lik = [sn2*np.trace(Q)]
                if covfunc.hyp:
                    for ii in range(len(covfunc.hyp)):
                        dnlZ.cov[ii] = old_div((Q*covfunc.getDerMatrix(x=x, mode='train', der=ii)).sum(),2.)
                if meanfunc.hyp:
                    for ii in range(len(meanfunc.hyp)):
                        dnlZ.mean[ii] = np.dot(-meanfunc.getDerMatrix(x, ii).T,alpha)
                        dnlZ.mean[ii] = dnlZ.mean[ii][0,0]
                return post, nlZ[0,0], dnlZ
            return post, nlZ[0,0]
        return post


class FITC_Exact(Inference):
    '''
    FITC approximation to the posterior Gaussian process. The function is
    equivalent to infExact with the covariance function:
    Kt = Q + G; G = diag(g); g = diag(K-Q);  Q = Ku' * inv(Quu) * Ku;
    where Ku and Kuu are covariances w.r.t. to inducing inputs xu, snu2 = sn2/1e6
    is the noise of the inducing inputs and Quu = Kuu + snu2*eye(nu).
    '''
    def __init__(self):
        self.name = 'FICT exact inference'

    def evaluate(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        if not isinstance(likfunc, lik.Gauss):                  # NOTE: no explicit call to likGauss
            raise Exception ('Exact inference only possible with Gaussian likelihood')
        if not isinstance(covfunc, cov.FITCOfKernel):
            raise Exception('Only covFITC supported.')          # check cov

        diagK,Kuu,Ku = covfunc.getCovMatrix(x=x, mode='train')  # evaluate covariance matrix
        m  = meanfunc.getMean(x)                                # evaluate mean vector
        n, D = x.shape
        nu = Kuu.shape[0]

        sn2   = np.exp(2*likfunc.hyp[0])                         # noise variance of likGauss
        snu2  = 1.e-6*sn2                                        # hard coded inducing inputs noise
        #Luu   = np.linalg.cholesky(Kuu+snu2*np.eye(nu)).T       # Kuu + snu2*I = Luu'*Luu
        Luu   = jitchol(Kuu+snu2*np.eye(nu)).T                   # Kuu + snu2*I = Luu'*Luu
        V     = np.linalg.solve(Luu.T,Ku)                        # V = inv(Luu')*Ku => V'*V = Q        
        
        g_sn2 = diagK + sn2 - np.array([(V*V).sum(axis=0)]).T    # g + sn2 = diag(K) + sn2 - diag(Q)
        #Lu    = np.linalg.cholesky(np.eye(nu) + np.dot(V/np.tile(g_sn2.T,(nu,1)),V.T)).T  # Lu'*Lu=I+V*diag(1/g_sn2)*V'
        Lu    = jitchol(np.eye(nu) + np.dot(old_div(V,np.tile(g_sn2.T,(nu,1))),V.T)).T  # Lu'*Lu=I+V*diag(1/g_sn2)*V'
        r     = old_div((y-m),np.sqrt(g_sn2))
        be    = np.linalg.solve(Lu.T,np.dot(V,old_div(r,np.sqrt(g_sn2))))
        iKuu  = solve_chol(Luu,np.eye(nu))                       # inv(Kuu + snu2*I) = iKuu

        post = postStruct()
        post.alpha = np.linalg.solve(Luu,np.linalg.solve(Lu,be)) # return the posterior parameters
        post.L  = solve_chol(np.dot(Lu,Luu),np.eye(nu)) - iKuu   # Sigma-inv(Kuu)
        post.sW = old_div(np.ones((n,1)),np.sqrt(sn2))                    # unused for FITC prediction  with gp.m

        if nargout>1:                                            # do we want the marginal likelihood
            nlZ = np.log(np.diag(Lu)).sum() + old_div((np.log(g_sn2).sum() + n*np.log(2*np.pi) + np.dot(r.T,r) - np.dot(be.T,be)),2.)
            if nargout>2:                                        # do we want derivatives?
                dnlZ = dnlZStruct(meanfunc, covfunc, likfunc)    # allocate space for derivatives
                al = old_div(r,np.sqrt(g_sn2)) - old_div(np.dot(V.T,np.linalg.solve(Lu,be)),g_sn2) # al = (Kt+sn2*eye(n))\y
                B = np.dot(iKuu,Ku)
                w = np.dot(B,al)
                W = np.linalg.solve(Lu.T,old_div(V,np.tile(g_sn2.T,(nu,1))))
                for ii in range(len(covfunc.hyp)):
                    [ddiagKi,dKuui,dKui] = covfunc.getDerMatrix(x=x, mode='train', der=ii)    # eval cov deriv
                    R = 2.*dKui-np.dot(dKuui,B)
                    v = ddiagKi - np.array([(R*B).sum(axis=0)]).T          # diag part of cov deriv
                    dnlZ.cov[ii] = old_div(( np.dot(ddiagKi.T,old_div(1.,g_sn2)) + np.dot(w.T,(np.dot(dKuui,w)-2.*np.dot(dKui,al))) \
                                   - np.dot(al.T,(v*al)) - np.dot(np.array([(W*W).sum(axis=0)]),v) - (np.dot(R,W.T)*np.dot(B,W.T)).sum() ),2.)
                    dnlZ.cov[ii] = dnlZ.cov[ii][0,0]
                dnlZ.lik = sn2*((old_div(1.,g_sn2)).sum() - (np.array([(W*W).sum(axis=0)])).sum() - np.dot(al.T,al))
                dKuui = 2*snu2
                R = -dKuui*B
                v = -np.array([(R*B).sum(axis=0)]).T     # diag part of cov deriv
                dnlZ.lik += old_div((np.dot(w.T,np.dot(dKuui,w)) -np.dot(al.T,(v*al)) \
                                 - np.dot(np.array([(W*W).sum(axis=0)]),v) - (np.dot(R,W.T)*np.dot(B,W.T)).sum() ),2.)
                dnlZ.lik = list(dnlZ.lik[0])
                for ii in range(len(meanfunc.hyp)):
                    dnlZ.mean[ii] = np.dot(-meanfunc.getDerMatrix(x, ii).T, al)
                    dnlZ.mean[ii] = dnlZ.mean[ii][0,0]

                return post, nlZ[0,0], dnlZ
            return post, nlZ[0,0]
        return post



class Laplace(Inference):
    '''
    Laplace's Approximation to the posterior Gaussian process.
    '''
    def __init__(self):
        self.last_alpha = None

    def evaluate(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        tol = 1e-6                           # tolerance for when to stop the Newton iterations
        smax = 2; Nline = 20; thr = 1e-4     # line search parameters
        maxit = 20                           # max number of Newton steps in f
        inffunc = self
        K = covfunc.getCovMatrix(x=x, mode='train')    # evaluate the covariance matrix
        m = meanfunc.getMean(x)              # evaluate the mean vector
        n, D = x.shape
        Psi_old = np.inf                     # make sure while loop starts by the largest old objective val
        if self.last_alpha is None:          # find a good starting point for alpha and f
            alpha = np.zeros((n,1))
            f = np.dot(K,alpha) + m          # start at mean if sizes not match
            vargout = likfunc.evaluate(y, f, None, inffunc, None, 3)
            lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]
            W= -d2lp
            Psi_new = -lp.sum()
        else:
            alpha = self.last_alpha
            f = np.dot(K,alpha) + m                       # try last one
            vargout = likfunc.evaluate(y, f, None, inffunc, None, 3)
            lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]
            W= -d2lp
            Psi_new = old_div(np.dot(alpha.T,(f-m)),2.) - lp.sum() # objective for last alpha
            vargout = - likfunc.evaluate(y, m, None, inffunc, None, 1)
            Psi_def =  vargout[0]                         # objective for default init f==m
            if Psi_def < Psi_new:                         # if default is better, we use it
                alpha = np.zeros((n,1))
                f = np.dot(K,alpha) + m
                vargout = likfunc.evaluate(y, f, None, inffunc, None, 3)
                lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]
                W=-d2lp; Psi_new = -lp.sum()
        isWneg = np.any(W<0)       # flag indicating whether we found negative values of W
        it = 0                     # this happens for the Student's t likelihood

        while (Psi_old - Psi_new > tol) and it<maxit:      # begin Newton
            Psi_old = Psi_new; it += 1
            if isWneg:                   # stabilise the Newton direction in case W has negative values
                W = np.maximum(W,0)      # stabilise the Hessian to guarantee postive definiteness
                tol = 1e-10              # increase accuracy to also get the derivatives right
            #sW = np.sqrt(W); L = np.linalg.cholesky(np.eye(n) + np.dot(sW,sW.T)*K).T
            sW = np.sqrt(W); L = jitchol(np.eye(n) + np.dot(sW,sW.T)*K).T
            b = W*(f-m) + dlp;
            dalpha = b - sW*solve_chol(L,sW*np.dot(K,b)) - alpha
            vargout = brentmin(0,smax,Nline,thr,self._Psi_line,4,dalpha,alpha,K,m,likfunc,y,inffunc)
            s = vargout[0]
            Psi_new = vargout[1]
            Nfun = vargout[2]
            alpha = vargout[3]
            f = vargout[4]
            dlp = vargout[5]
            W = vargout[6]
            isWneg = np.any(W<0)
        self.last_alpha = alpha                 # remember for next call
        vargout = likfunc.evaluate(y,f,None,inffunc,None,4)
        lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]; d3lp = vargout[3]
        W = -d2lp; isWneg = np.any(W<0)
        post = postStruct()
        post.alpha = alpha                      # return the posterior parameters
        post.sW = np.sqrt(np.abs(W))*np.sign(W) # preserve sign in case of negative
        if isWneg:
            [ldA,iA,post.L] = self._logdetA(K,W,3)
            nlZ = old_div(np.dot(alpha.T,(f-m)),2.) - lp.sum() + old_div(ldA,2.)
            nlZ = nlZ[0]
        else:
            sW = post.sW
            #post.L = np.linalg.cholesky(np.eye(n)+np.dot(sW,sW.T)*K).T
            post.L = jitchol(np.eye(n)+np.dot(sW,sW.T)*K).T
            nlZ = old_div(np.dot(alpha.T,(f-m)),2.) + (np.log(np.diag(post.L))-np.reshape(lp,(lp.shape[0],))).sum()
            nlZ = nlZ[0]
        if nargout>2:                                           # do we want derivatives?
            dnlZ = dnlZStruct(meanfunc, covfunc, likfunc)       # allocate space for derivatives
            if isWneg:                                          # switch between Cholesky and LU decomposition mode
                Z = -post.L                                     # inv(K+inv(W))
                g = old_div(np.atleast_2d((iA*K).sum(axis=1)).T,2)      # deriv. of ln|B| wrt W; g = diag(inv(inv(K)+diag(W)))/2
            else:
                Z = np.tile(sW,(1,n))*solve_chol(post.L,np.diag(np.reshape(sW,(sW.shape[0],)))) #sW*inv(B)*sW=inv(K+inv(W))
                C = np.linalg.solve(post.L.T,np.tile(sW,(1,n))*K)            # deriv. of ln|B| wrt W
                g = old_div(np.atleast_2d((np.diag(K)-(C**2).sum(axis=0).T)).T,2.)   # g = diag(inv(inv(K)+W))/2
            dfhat = g* d3lp                 # deriv. of nlZ wrt. fhat: dfhat=diag(inv(inv(K)+W)).*d3lp/2
            for ii in range(len(covfunc.hyp)):                               # covariance hypers
                dK = covfunc.getDerMatrix(x=x, mode='train', der=ii)
                dnlZ.cov[ii] = old_div((Z*dK).sum(),2.) - old_div(np.dot(alpha.T,np.dot(dK,alpha)),2.)   # explicit part
                b = np.dot(dK,dlp)                                           # b-K*(Z*b) = inv(eye(n)+K*diag(W))*b
                dnlZ.cov[ii] -= np.dot(dfhat.T,b-np.dot(K,np.dot(Z,b)))      # implicit part
                dnlZ.cov[ii] = dnlZ.cov[ii][0,0]
            for ii in range(len(likfunc.hyp)):                               # likelihood hypers
                [lp_dhyp,dlp_dhyp,d2lp_dhyp] = likfunc.evaluate(y,f,None,inffunc,ii,3)
                dnlZ.lik[ii] = -np.dot(g.T,d2lp_dhyp) - lp_dhyp.sum()        # explicit part
                b = np.dot(K,dlp_dhyp)                                       # b-K*(Z*b) = inv(eye(n)+K*diag(W))*b
                dnlZ.lik[ii] -= np.dot(dfhat.T,b-np.dot(K,np.dot(Z,b)))      # implicit part
                dnlZ.lik[ii] = dnlZ.lik[ii][0,0]
            for ii in range(len(meanfunc.hyp)):                              # mean hypers
                dm = meanfunc.getDerMatrix(x, ii)
                dnlZ.mean[ii] = -np.dot(alpha.T,dm)                          # explicit part
                dnlZ.mean[ii] -= np.dot(dfhat.T,dm-np.dot(K,np.dot(Z,dm)))   # implicit part
                dnlZ.mean[ii] = dnlZ.mean[ii][0,0]
            return post,nlZ[0],dnlZ
        else:
            return post, nlZ[0]



class FITC_Laplace(Inference):
    '''
    FITC-Laplace approximation to the posterior Gaussian process. The function is
    equivalent to Laplace with the covariance function:
    Kt = Q + G; G = diag(g); g = diag(K-Q);  Q = Ku' * inv(Kuu + snu2 * eye(nu)) * Ku;
    where Ku and Kuu are covariances w.r.t. to inducing inputs xu and
    snu2 = sn2/1e6 is the noise of the inducing inputs. We fixed the standard
    deviation of the inducing inputs snu to be a one per mil of the measurement
    noise's standard deviation sn. In case of a likelihood without noise
    parameter sn2, we simply use snu2 = 1e-6.
    '''
    def __init__(self):
        self.last_alpha = None

    def evaluate(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        if not isinstance(covfunc, cov.FITCOfKernel):
            raise Exception('Only covFITC supported.')
        tol = 1e-6                             # tolerance for when to stop the Newton iterations
        smax = 2; Nline = 100; thr = 1e-4      # line search parameters
        maxit = 20                             # max number of Newton steps in f
        inffunc = Laplace()
        diagK,Kuu,Ku = covfunc.getCovMatrix(x=x, mode='train')      # evaluate the covariance matrix
        m = meanfunc.getMean(x)                # evaluate the mean vector
        if likfunc.hyp:                        # hard coded inducing inputs noise
            sn2  = np.exp(2.*likfunc.hyp[-1])
            snu2 = 1.e-6*sn2                   # similar to infFITC
        else:
            snu2 = 1.e-6

        n, D = x.shape
        nu = Kuu.shape[0]
        rot180   = lambda A: np.rot90(np.rot90(A))      # little helper functions
        #chol_inv = lambda A: np.linalg.solve( rot180( np.linalg.cholesky(rot180(A)) ),np.eye(nu)) # chol(inv(A))
        chol_inv = lambda A: rot180( np.linalg.solve(jitchol(rot180(A)), np.eye(nu)) ) # chol(inv(A))
        
        R0 = chol_inv(Kuu+snu2*np.eye(nu))              # initial R, used for refresh O(nu^3)
        V  = np.dot(R0,Ku); d0 = diagK - np.array([(V*V).sum(axis=0)]).T     # initial d, needed

        Psi_old = np.inf                    # make sure while loop starts by the largest old objective val
        if self.last_alpha is None:         # find a good starting point for alpha and f
            alpha = np.zeros((n,1))
            f = self._mvmK(alpha,V,d0) + m   # start at mean if sizes not match
            vargout = likfunc.evaluate(y, f, None, inffunc, None, 3)
            lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]
            W=-d2lp; Psi_new = -lp.sum()
        else:
            alpha = self.last_alpha
            f = self._mvmK(alpha,V,d0) + m                           # try last one
            vargout = likfunc.evaluate(y, f, None, inffunc, None, 3)
            lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]
            W=-d2lp
            Psi_new = old_div(np.dot(alpha.T,(f-m)),2.) - lp.sum()           # objective for last alpha
            vargout = - likfunc.evaluate(y, m, None, inffunc, None, 1)
            Psi_def =  vargout[0]                                   # objective for default init f==m
            if Psi_def < Psi_new:                                   # if default is better, we use it
                alpha = np.zeros((n,1))
                f = self._mvmK(alpha,V,d0) + m
                vargout = likfunc.evaluate(y, f, None, inffunc, None, 3)
                lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]
                W=-d2lp; Psi_new = -lp.sum()

        isWneg = np.any(W<0)                # flag indicating whether we found negative values of W
        it = 0                              # this happens for the Student's t likelihood

        while (Psi_old - Psi_new > tol) and it<maxit:               # begin Newton
            Psi_old = Psi_new
            it += 1
            if isWneg:                      # stabilise the Newton direction in case W has negative values
                W = np.maximum(W,0)         # stabilise the Hessian to guarantee postive definiteness
                tol = 1e-8                  # increase accuracy to also get the derivatives right
            b = W*(f-m) + dlp; dd = old_div(1,(1+W*d0))
            RV = np.dot( chol_inv( np.eye(nu) + np.dot(V*np.tile((W*dd).T,(nu,1)),V.T)),V )
            dalpha = dd*b - (W*dd)*np.dot(RV.T,np.dot(RV,(dd*b))) - alpha # Newt dir + line search
            vargout = brentmin(0,smax,Nline,thr,self._Psi_lineFITC,4,dalpha,alpha,V,d0,m,likfunc,y,inffunc)
            s = vargout[0]; Psi_new = vargout[1]; Nfun = vargout[2]; alpha = vargout[3]
            f = vargout[4]; dlp = vargout[5]; W = vargout[6]
            isWneg = np.any(W<0)

        self.last_alpha = alpha                                     # remember for next call
        vargout = likfunc.evaluate(y,f,None,inffunc,None,4)
        lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]; d3lp = vargout[3]

        W=-d2lp; isWneg = np.any(W<0)
        post = postStruct()
        post.alpha = np.dot(R0.T,np.dot(V,alpha))                   # return the posterior parameters
        post.sW = np.sqrt(np.abs(W))*np.sign(W)                     # preserve sign in case of negative
        dd = old_div(1,(1+d0*W))                                             # temporary variable O(n)
        A = np.eye(nu) + np.dot(V*np.tile((W*dd).T,(nu,1)),V.T)     # temporary variable O(n*nu^2)
        R0tV = np.dot(R0.T,V); B = R0tV*np.tile((W*dd).T,(nu,1))    # temporary variables O(n*nu^2)
        post.L = -np.dot(B,R0tV.T)                                  # L = -R0'*V*inv(Kt+diag(1./ttau))*V'*R0, first part
        if np.any(1+d0*W<0):
            raise Exception('W is too negative; nlZ and dnlZ cannot be computed.')
        #nlZ = np.dot(alpha.T,(f-m))/2. - lp.sum() - np.log(dd).sum()/2. + \
        #    np.log(np.diag(np.linalg.cholesky(A).T)).sum()
        nlZ = old_div(np.dot(alpha.T,(f-m)),2.) - lp.sum() - old_div(np.log(dd).sum(),2.) + \
            np.log(np.diag(jitchol(A).T)).sum()
        RV = np.dot(chol_inv(A),V)
        RVdd = RV * np.tile((W*dd).T,(nu,1))                        # RVdd needed for dnlZ
        B = np.dot(B,RV.T)
        post.L += np.dot(B,B.T)

        if nargout>2:                                               # do we want derivatives?
            dnlZ = dnlZStruct(meanfunc, covfunc, likfunc)           # allocate space for derivatives
            [d,P,R] = self._fitcRefresh(d0,Ku,R0,V,W)                # g = diag(inv(inv(K)+W))/2
            g = old_div(d,2) + 0.5*np.atleast_2d((np.dot(np.dot(R,R0),P)**2).sum(axis=0)).T
            t = old_div(W,(1+W*d0))

            dfhat = g*d3lp  # deriv. of nlZ wrt. fhat: dfhat=diag(inv(inv(K)+W)).*d3lp/2
            for ii in range(len(covfunc.hyp)):                      # covariance hypers
                ddiagK,dKuu,dKu = covfunc.getDerMatrix(x=x, mode='train', der=ii)      # eval cov derivatives
                dA = 2.*dKu.T-np.dot(R0tV.T,dKuu)                   # dQ = dA*R0tV
                w = np.atleast_2d((dA*R0tV.T).sum(axis=1)).T        # w = diag(dQ)
                v = ddiagK-w                                        # v = diag(dK)-diag(dQ);
                dnlZ.cov[ii] = np.dot(ddiagK.T,t) - np.dot((RVdd*RVdd).sum(axis=0),v)   # explicit part
                dnlZ.cov[ii] -= (np.dot(RVdd,dA)*np.dot(RVdd,R0tV.T)).sum()             # explicit part
                dnlZ.cov[ii] = 0.5*dnlZ.cov[ii] - old_div(np.dot(alpha.T,np.dot(dA,np.dot(R0tV,alpha))+v*alpha),2.) # explicit
                b = np.dot(dA,np.dot(R0tV,dlp)) + v*dlp             # b-K*(Z*b) = inv(eye(n)+K*diag(W))*b
                KZb = self._mvmK(self._mvmZ(b,RVdd,t),V,d0)
                dnlZ.cov[ii] -= np.dot(dfhat.T,(b-KZb))             # implicit part
                dnlZ.cov[ii] = dnlZ.cov[ii][0,0]

            for ii in range(len(likfunc.hyp)):                      # likelihood hypers
                vargout = likfunc.evaluate(y,f,None,inffunc,ii,3)
                lp_dhyp = vargout[0]; dlp_dhyp = vargout[1]; d2lp_dhyp = vargout[2]
                dnlZ.lik[ii] = -np.dot(g.T,d2lp_dhyp) - lp_dhyp.sum() # explicit part
                b = self._mvmK(dlp_dhyp,V,d0)                          # implicit part
                dnlZ.lik[ii] -= np.dot(dfhat.T,b-self._mvmK(self._mvmZ(b,RVdd,t),V,d0))
                if ii == len(likfunc.hyp)-1:
                    # since snu2 is a fixed fraction of sn2, there is a covariance-like term
                    # in the derivative as well
                    snu = np.sqrt(snu2);
                    T = chol_inv(Kuu + snu2*np.eye(nu));
                    T = np.dot(T.T,np.dot(T,snu*Ku));
                    t = np.array([(T*T).sum(axis=0)]).T
                    z = np.dot(alpha.T,np.dot(T.T,np.dot(T,alpha))-t*alpha) - np.dot(np.array([(RVdd*RVdd).sum(axis=0)]),t)
                    z += (np.dot(RVdd,T.T)**2).sum()
                    b = old_div((t*dlp-np.dot(T.T,np.dot(T,dlp))),2.)
                    KZb = self._mvmK(self._mvmZ(b,RVdd,t),V,d0)
                    z -= np.dot(dfhat.T,b-KZb)
                    dnlZ.lik[ii] += z
                    dnlZ.lik[ii] = dnlZ.lik[ii][0,0]

            for ii in range(len(meanfunc.hyp)):                           # mean hypers
                dm = meanfunc.getDerMatrix(x, ii)
                dnlZ.mean[ii] = -np.dot(alpha.T,dm)                       # explicit part
                Zdm = self._mvmZ(dm,RVdd,t)
                dnlZ.mean[ii] -= np.dot(dfhat.T,(dm-self._mvmK(Zdm,V,d0))) # implicit part
                dnlZ.mean[ii] = dnlZ.mean[ii][0,0]

            return post,nlZ[0,0],dnlZ
        else:
            return post, nlZ[0,0]



class EP(Inference):
    '''
    Expectation Propagation approximation to the posterior Gaussian Process.
    '''
    def __init__(self):
        self.name = 'Expectation Propagation'
        self.last_ttau = None
        self.last_tnu = None
    def evaluate(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        tol = 1e-4; max_sweep = 10; min_sweep = 2 # tolerance to stop EP iterations
        n = x.shape[0]
        inffunc = self
        K = covfunc.getCovMatrix(x=x, mode='train') # evaluate the covariance matrix
        m = meanfunc.getMean(x)                   # evaluate the mean vector
        nlZ0 = -likfunc.evaluate(y, m, np.reshape(np.diag(K),(np.diag(K).shape[0],1)), inffunc).sum()
        if self.last_ttau is None:                # find starting point for tilde parameters
            ttau  = np.zeros((n,1))               # initialize to zero if we have no better guess
            tnu   = np.zeros((n,1))
            Sigma = K                             # initialize Sigma and mu, the parameters of ..
            mu    = np.zeros((n,1))               # .. the Gaussian posterior approximation
            nlZ   = nlZ0
        else:
            ttau = self.last_ttau                 # try the tilde values from previous call
            tnu  = self.last_tnu
            Sigma, mu, nlZ, L = self._epComputeParams(K, y, ttau, tnu, likfunc, m, inffunc)
            if nlZ > nlZ0:                        # if zero is better ..
                ttau = np.zeros((n,1))            # .. then initialize with zero instead
                tnu  = np.zeros((n,1))
                Sigma = K                         # initialize Sigma and mu, the parameters of ..
                mu = np.zeros((n,1))              # .. the Gaussian posterior approximation
                nlZ = nlZ0
        nlZ_old = np.inf; sweep = 0               # converged, max. sweeps or min. sweeps?
        while (np.abs(nlZ-nlZ_old) > tol and sweep < max_sweep) or (sweep < min_sweep):
            nlZ_old = nlZ; sweep += 1
            rperm = range(n)                     # randperm(n)
            for ii in rperm:                      # iterate EP updates (in random order) over examples
                tau_ni = old_div(1,Sigma[ii,ii]) - ttau[ii]#  first find the cavity distribution ..
                nu_ni  = old_div(mu[ii],Sigma[ii,ii]) + m[ii]*tau_ni - tnu[ii]    # .. params tau_ni and nu_ni
                # compute the desired derivatives of the indivdual log partition function
                lZ,dlZ,d2lZ = likfunc.evaluate(y[ii], old_div(nu_ni,tau_ni), old_div(1,tau_ni), inffunc, None, 3)
                ttau_old = copy(ttau[ii])         # then find the new tilde parameters, keep copy of old
                ttau[ii] = old_div(-d2lZ,(1.+old_div(d2lZ,tau_ni)))
                ttau[ii] = max(ttau[ii],0)        # enforce positivity i.e. lower bound ttau by zero
                tnu[ii]  = old_div(( dlZ + (m[ii]-old_div(nu_ni,tau_ni))*d2lZ ),(1.+old_div(d2lZ,tau_ni)))
                ds2 = ttau[ii] - ttau_old         # finally rank-1 update Sigma ..
                si  = np.reshape(Sigma[:,ii],(Sigma.shape[0],1))
                Sigma = Sigma - ds2/(1.+ds2*si[ii])*np.dot(si,si.T)   # takes 70# of total time
                mu = np.dot(Sigma,tnu)                                # .. and recompute mu
            # recompute since repeated rank-one updates can destroy numerical precision
            Sigma, mu, nlZ, L = self._epComputeParams(K, y, ttau, tnu, likfunc, m, inffunc)
        if sweep == max_sweep:
            pass
            # print '[warning] maximum number of sweeps reached in function infEP'
        self.last_ttau = ttau; self.last_tnu = tnu          # remember for next call
        sW = np.sqrt(ttau); alpha = tnu-sW*solve_chol(L,sW*np.dot(K,tnu))
        post = postStruct()
        post.alpha = alpha                                  # return the posterior params
        post.sW    = sW
        post.L     = L
        if nargout>2:                                       # do we want derivatives?
            dnlZ = dnlZStruct(meanfunc, covfunc, likfunc)   # allocate space for derivatives
            ssi  = np.sqrt(ttau)
            V = np.linalg.solve(L.T,np.tile(ssi,(1,n))*K)
            Sigma = K - np.dot(V.T,V)
            mu = np.dot(Sigma,tnu)
            Dsigma = np.reshape(np.diag(Sigma),(np.diag(Sigma).shape[0],1))
            tau_n = old_div(1,Dsigma)-ttau                           # compute the log marginal likelihood
            nu_n  = old_div(mu,Dsigma)-tnu                           # vectors of cavity parameters
            F = np.dot(alpha,alpha.T) - np.tile(sW,(1,n))* \
                solve_chol(L,np.diag(np.reshape(sW,(sW.shape[0],))))   # covariance hypers
            for jj in range(len(covfunc.hyp)):
                dK = covfunc.getDerMatrix(x=x, mode='train', der=jj)
                dnlZ.cov[jj] = old_div(-(F*dK).sum(),2.)
            for ii in range(len(likfunc.hyp)):
                dlik = likfunc.evaluate(y, old_div(nu_n,tau_n), old_div(1,tau_n), inffunc, ii)
                dnlZ.lik[ii] = -dlik.sum()
            junk,dlZ = likfunc.evaluate(y, old_div(nu_n,tau_n), old_div(1,tau_n), inffunc, None, 2) # mean hyps
            for ii in range(len(meanfunc.hyp)):
                dm = meanfunc.getDerMatrix(x, ii)
                dnlZ.mean[ii] = -np.dot(dlZ.T,dm)
                dnlZ.mean[ii] = dnlZ.mean[ii][0,0]
            return post, nlZ[0], dnlZ
        else:
            return post, nlZ[0]



class FITC_EP(Inference):
    '''
    FITC-EP approximation to the posterior Gaussian process. The function is
    equivalent to infEP with the covariance function:
    Kt = Q + G; G = diag(g); g = diag(K-Q);  Q = Ku' * inv(Kuu + snu2 * eye(nu)) * Ku;
    where Ku and Kuu are covariances w.r.t. to inducing inputs xu and
    snu2 = sn2/1e6 is the noise of the inducing inputs. We fixed the standard
    deviation of the inducing inputs snu to be a one per mil of the measurement
    noise's standard deviation sn. In case of a likelihood without noise
    parameter sn2, we simply use snu2 = 1e-6.
    For details, see The Generalized FITC Approximation, Andrew Naish-Guzman and
    Sean Holden, NIPS, 2007.
    '''
    def __init__(self):
        self.name = 'FITC Expectation Propagation'
        self.last_ttau = None
        self.last_tnu = None

    def evaluate(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        if not isinstance(covfunc, cov.FITCOfKernel):
            raise Exception('Only covFITC supported.')  # check cov
        tol = 1e-4; max_sweep = 10; min_sweep = 2       # tolerance to stop EP iterations
        inffunc = EP()

        diagK,Kuu,Ku = covfunc.getCovMatrix(x=x, mode='train')  # evaluate the covariance matrix
        m = meanfunc.getMean(x)                         # evaluate the mean vector

        if likfunc.hyp:                                 # hard coded inducing inputs noise
            sn2  = np.exp(2.*likfunc.hyp[-1])
            snu2 = 1.e-6*sn2                            # similar to infFITC
        else:
            snu2 = 1.e-6

        n, D = x.shape; nu = Kuu.shape[0]
        rot180   = lambda A: np.rot90(np.rot90(A))      # little helper functions
        #chol_inv = lambda A: np.linalg.solve( rot180( np.linalg.cholesky(rot180(A)) ),np.eye(nu)) # chol(inv(A))
        chol_inv = lambda A: rot180( np.linalg.solve(jitchol(rot180(A)), np.eye(nu)) ) # chol(inv(A))

        R0 = chol_inv(Kuu+snu2*np.eye(nu))              # initial R, used for refresh O(nu^3)
        V  = np.dot(R0,Ku); d0 = diagK - np.array([(V*V).sum(axis=0)]).T # initial d, needed for refresh O(n*nu^2)

        # A note on naming: variables are given short but descriptive names in
        # accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
        # and s2 are mean and variance, nu and tau are natural parameters. A leading t
        # means tilde, a subscript _ni means "not i" (for cavity parameters), or _n
        # for a vector of cavity parameters.

        # marginal likelihood for ttau = tnu = zeros(n,1); equals n*log(2) for likCum*
        nlZ0 = -1.* likfunc.evaluate(y, m, np.reshape(diagK,(diagK.shape[0],1)), inffunc).sum()
        if self.last_ttau is None:                      # find starting point for tilde parameters
            ttau  = np.zeros((n,1))                     # initialize to zero if we have no better guess
            tnu   = np.zeros((n,1))
            [d,P,R,nn,gg] = self._epfitcRefresh(d0,Ku,R0,V,ttau,tnu)   # compute initial repres.
            nlZ = nlZ0
        else:
            ttau = self.last_ttau                       # try the tilde values from previous call
            tnu  = self.last_tnu
            [d,P,R,nn,gg] = self._epfitcRefresh(d0,Ku,R0,V,ttau,tnu) # compute initial repres.
            nlZ = self._epfitcZ(d,P,R,nn,gg,ttau,tnu,d0,R0,Ku,y,likfunc,m,inffunc)[0]
            if nlZ > nlZ0:                              # if zero is better ..
                ttau = np.zeros((n,1))                  # .. then initialize with zero instead
                tnu  = np.zeros((n,1))
                [d,P,R,nn,gg] = self._epfitcRefresh(d0,Ku,R0,V,ttau,tnu) # initial repres.
                nlZ = nlZ0

        nlZ_old = np.inf; sweep = 0                     # converged, max. sweeps or min. sweeps?
        while (np.abs(nlZ-nlZ_old) > tol and sweep < max_sweep) or (sweep < min_sweep):
            nlZ_old = nlZ
            sweep += 1
            rperm = list(range(n))                            # randperm(n)
            for ii in rperm:                            # iterate EP updates (in random order) over examples
                p_i = np.reshape(P[:,ii],(P.shape[0],1))
                t = np.dot(R,np.dot(R0,p_i))            # temporary variables
                sigma_i = d[ii] + np.dot(t.T,t); mu_i = nn[ii] + np.dot(p_i.T,gg) # post moments O(nu^2)
                tau_ni = old_div(1,sigma_i) - ttau[ii]                   #  first find the cavity distribution ..
                nu_ni  = old_div(mu_i,sigma_i) + m[ii]*tau_ni - tnu[ii]  # .. params tau_ni and nu_ni
                # compute the desired derivatives of the indivdual log partition function
                vargout = likfunc.evaluate(y[ii], old_div(nu_ni,tau_ni), old_div(1,tau_ni), inffunc, None, 3)
                lZ = vargout[0]; dlZ = vargout[1]; d2lZ = vargout[2]
                ttau_i = old_div(-d2lZ,(1.+old_div(d2lZ,tau_ni)))
                ttau_i = max(ttau_i,0)                  # enforce positivity i.e. lower bound ttau by zero
                tnu_i  = old_div(( dlZ + (m[ii]-old_div(nu_ni,tau_ni))*d2lZ ),(1.+old_div(d2lZ,tau_ni)))
                [d,P[:,ii],R,nn,gg,ttau,tnu] = self._epfitcUpdate(d,P[:,ii],R,nn,gg,ttau,tnu,ii,ttau_i,tnu_i,m,d0,Ku,R0)# update representation

            # recompute since repeated rank-one updates can destroy numerical precision
            [d,P,R,nn,gg] = self._epfitcRefresh(d0,Ku,R0,V,ttau,tnu)
            [nlZ,nu_n,tau_n] = self._epfitcZ(d,P,R,nn,gg,ttau,tnu,d0,R0,Ku,y,likfunc,m,inffunc)
        if sweep == max_sweep:
            pass
            # print '[warning] maximum number of sweeps reached in function infEP'

        self.last_ttau = ttau
        self.last_tnu = tnu       # remember for next call
        post = postStruct()
        post.sW = np.sqrt(ttau)   # unused for FITC_EP prediction with gp.m
        dd = old_div(1,(d0+old_div(1,ttau)))
        alpha = tnu/ttau*dd
        RV = np.dot(R,V)
        R0tV = np.dot(R0.T,V)
        alpha = alpha - np.dot(RV.T,np.dot(RV,alpha))*dd      # long alpha vector for ordinary infEP
        post.alpha = np.dot(R0tV,alpha)                       # alpha = R0'*V*inv(Kt+diag(1./ttau))*(tnu./ttau)
        B = R0tV*np.tile(dd.T,(nu,1)); L = np.dot(B,R0tV.T); B = np.dot(B,RV.T)
        post.L = np.dot(B,B.T) - L                            # L = -R0'*V*inv(Kt+diag(1./ttau))*V'*R0

        if nargout>2:                                         # do we want derivatives?
            dnlZ = dnlZStruct(meanfunc, covfunc, likfunc)     # allocate space for derivatives
            RVdd = RV*np.tile(dd.T,(nu,1))
            for ii in range(len(covfunc.hyp)):
                ddiagK,dKuu,dKu = covfunc.getDerMatrix(x=x, mode='train', der=ii)
                dA = 2*dKu.T - np.dot(R0tV.T,dKuu)            # dQ = dA*R0tV
                w = np.atleast_2d((dA*R0tV.T).sum(axis=1)).T  # w = diag(dQ)
                v = ddiagK - w                                # v = diag(dK)-diag(dQ)
                z = np.dot(dd.T,(v+w)) - np.dot(np.atleast_2d((RVdd*RVdd).sum(axis=0)), v) \
                       - (np.dot(RVdd,dA).T * np.dot(R0tV,RVdd.T)).sum()
                dnlZ.cov[ii] = old_div((z - np.dot(alpha.T,(alpha*v)) - np.dot(np.dot(alpha.T,dA),np.dot(R0tV,alpha))),2.)
                dnlZ.cov[ii] = dnlZ.cov[ii][0,0]
            for ii in range(len(likfunc.hyp)):                # likelihood hypers
                dlik = likfunc.evaluate(y, old_div(nu_n,tau_n)+m, old_div(1,tau_n), inffunc, ii, 1)
                dnlZ.lik[ii] = -dlik.sum()
                if ii == len(likfunc.hyp)-1:
                    # since snu2 is a fixed fraction of sn2, there is a covariance-like term
                    # in the derivative as well
                    v = np.atleast_2d((R0tV*R0tV).sum(axis=0)).T
                    z = (np.dot(RVdd,R0tV.T)**2).sum()  - np.dot(np.atleast_2d((RVdd*RVdd).sum(axis=0)),v)
                    z = z + np.dot(post.alpha.T,post.alpha) - np.dot(alpha.T,(v*alpha))
                    dnlZ.lik[ii] += snu2*z
                    dnlZ.lik[ii] = dnlZ.lik[ii][0,0]
            [junk,dlZ] = likfunc.evaluate(y, old_div(nu_n,tau_n), old_div(1,tau_n), inffunc, None, 2) # mean hyps
            for ii in range(len(meanfunc.hyp)):
                dm = meanfunc.getDerMatrix(x, ii)
                dnlZ.mean[ii] = -np.dot(dlZ.T,dm)
                dnlZ.mean[ii] = dnlZ.mean[ii][0,0]

            return post, nlZ[0,0], dnlZ
        else:
            return post, nlZ[0,0]



if __name__ == '__main__':
    pass


