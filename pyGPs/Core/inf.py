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

# Inference methods: Compute the (approximate) posterior for a Gaussian process.
# Methods currently implemented include:
#
#   infExact         Exact inference (only possible with Gaussian likelihood)
#   infLaplace       Laplace's Approximation
#   infEP            Expectation Propagation
#   infVB            [NOT IMPLEMENTED!] Variational Bayes Approximation 
#
#   infFITC          Large scale regression with approximate covariance matrix
#   infFITC_Laplace  Large scale inference  with approximate covariance matrix
#   infFITC_EP       Large scale inference  with approximate covariance matrix
#
#   infMCMC     [NOT IMPLEMENTED!]
#               Markov Chain Monte Carlo and Annealed Importance Sampling
#               We offer two samplers.
#                 - hmc: Hybrid Monte Carlo
#                 - ess: Elliptical Slice Sampling
#               No derivatives w.r.t. to hyperparameters are provided.
#
#   infLOO      [NOT IMPLEMENTED!]
#               Leave-One-Out predictive probability and Least-Squares Approxim.
#
# The interface to the approximation methods is the following:
#
# post nlZ dnlZ = inf.proceed(cov, lik, x, y)
#
# where:
#   INPUT:
#   cov     name of the covariance function (see covFunctions.m)
#   lik     name of the likelihood function (see likFunctions.m)
#   x       n by D matrix of training inputs 
#   y       1d array (of size n) of targets
#
#   OUTPUT:
#   post    struct representation of the (approximate) posterior containing: 
#           alpha   1d array containing inv(K)*m, 
#                   where K is the prior covariance matrix and m the approx posterior mean
#           sW      1d array containing diagonal of sqrt(W)
#                   the approximate posterior covariance matrix is inv(inv(K)+W)
#           L       2d array, L = chol(sW*K*sW+identity(n))
#   nlZ     returned value of the negative log marginal likelihood
#   dnlZ    1d array of partial derivatives of the negative log marginal likelihood
#           w.r.t. each hyperparameter
#
# Usually, the approximate posterior to be returned admits the form
# N(m=K*alpha, V=inv(inv(K)+W)), where alpha is a vector and W is diagonal;
# if not, then L contains instead -inv(K+inv(W)), and sW is unused.
#
# For more information on the individual approximation methods and their
# implementations, see the respective inf* function below. See also gp.py
#
#
# @author: Shan Huang (last update Sep.2013)
# This is a object-oriented python implementation of gpml functionality 
# (Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18).
# based on the functional-version of python implementation
# (Copyright (c) by Marion Neumann and Daniel Marthaler, 20/05/2013)
# 
#
# Copyright (c) by Marion Neumann and Shan Huang, Sep.2013


import numpy as np
import lik, cov
from copy import copy,deepcopy
from tools import solve_chol,brentmin,cholupdate

np.seterr(all='ignore')

class postStruct(object):
    def __init__(self):
        self.alpha = np.array([])
        self.L     = np.array([])
        self.sW    = np.array([])

class dnlZStruct(object):
    def __init__(self, m, c, l):
        self.mean = []
        self.cov = []
        self.lik = []
        if m.hyp:
            self.mean = [0 for i in xrange(len(m.hyp))]
        if c.hyp:
            self.cov  = [0 for i in xrange(len(c.hyp))]
        if l.hyp:
            self.lik  = [0 for i in xrange(len(l.hyp))]        

class Inference(object):
    '''
    Base class for inference. Defined several tool methods in it.
    '''
    def __init__(self):
        pass

    def proceed(self):
        # return [post nlZ dnlZ]
        pass

    def epComputeParams(self, K, y, ttau, tnu, likfunc, m, inffunc):
        n     = len(y)                                                # number of training cases
        ssi   = np.sqrt(ttau)                                         # compute Sigma and mu
        L     = np.linalg.cholesky(np.eye(n)+np.dot(ssi,ssi.T)*K).T   # L'*L=B=eye(n)+sW*K*sW
        V     = np.linalg.solve(L.T,np.tile(ssi,(1,n))*K)
        Sigma = K - np.dot(V.T,V)
        mu    = np.dot(Sigma,tnu)
        Dsigma = np.reshape(np.diag(Sigma),(np.diag(Sigma).shape[0],1)) 
        tau_n = 1/Dsigma - ttau               # compute the log marginal likelihood
        nu_n  = mu/Dsigma-tnu + m*tau_n       # vectors of cavity parameters
        lZ    = likfunc.proceed(y, nu_n/tau_n, 1/tau_n, inffunc)
        nlZ   = np.log(np.diag(L)).sum() - lZ.sum() - np.dot(tnu.T,np.dot(Sigma,tnu))/2  \
                - np.dot((nu_n-m*tau_n).T,((ttau/tau_n*(nu_n-m*tau_n)-2*tnu) / (ttau+tau_n)))/2 \
                + (tnu**2/(tau_n+ttau)).sum()/2.- np.log(1.+ttau/tau_n).sum()/2.
        return Sigma, mu, nlZ[0], L
    
    def logdetA(self,K,w,nargout):
        # Compute the log determinant ldA and the inverse iA of a square nxn matrix
        # A = eye(n) + K*diag(w) from its LU decomposition; for negative definite A, we 
        # return ldA = Inf. We also return mwiA = -diag(w)*inv(A).
        # [ldA,iA,mwiA] = logdetA(K,w)
        n = K.shape[0]
        assert(K.shape[0] == K.shape[1])
        A = np.eye(n) + K*np.tile(w.T,(n,1))
        [L,U,P] = np.linalg.lu(A)    # compute LU decomposition, A = P'*L*U
        u = np.diag(U)           
        signU = np.prod(np.sign(u))  # sign of U
        detP = 1                     # compute sign (and det) of the permutation matrix P
        p = np.dot(P,np.array(range(n)).T)
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
    
    def Psi_line(self,s,dalpha,alpha,K,m,likfunc,y,inffunc):
        # criterion Psi at alpha + s*dalpha for line search
        # [Psi,alpha,f,dlp,W] = Psi_line(s,dalpha,alpha,hyp,K,m,lik,y,inf)
        alpha = alpha + s*dalpha
        f = np.dot(K,alpha) + m
        [lp,dlp,d2lp] = likfunc.proceed(y,f,None,inffunc,None,3) 
        W = -d2lp
        Psi = np.dot(alpha.T,(f-m))/2. - lp.sum()
        return Psi[0],alpha,f,dlp,W  
    
    def epfitcZ(self,d,P,R,nn,gg,ttau,tnu,d0,R0,P0,y,likfunc,m,inffunc):
        # compute the marginal likelihood approximation
        # effort is O(n*nu^2) provided that nu<n
        T = np.dot(np.dot(R,R0),P)              # temporary variable
        diag_sigma = d + np.array([(T*T).sum(axis=0)]).T 
        mu = nn + np.dot(P.T,gg)                # post moments O(n*nu^2)
        tau_n = 1./diag_sigma-ttau              # compute the log marginal likelihood
        nu_n  = mu/diag_sigma - tnu + m*tau_n   # vectors of cavity parameters
        lZ = likfunc.proceed(y, nu_n/tau_n, 1/tau_n, inffunc, None, 1)
        nu = gg.shape[0]
        U = np.dot(R0,P0).T*np.tile(1/np.sqrt(d0+1/ttau),(1,nu))
        L = np.linalg.cholesky(np.eye(nu)+np.dot(U.T,U)).T
        ld = 2.*np.log(np.diag(L)).sum() + (np.log(d0+1/ttau)).sum() + (np.log(ttau)).sum()
        t = np.dot(T,tnu); tnu_Sigma_tnu = np.dot(tnu.T,(d*tnu)) + np.dot(t.T,t)
        nlZ = ld/2. - lZ.sum() -tnu_Sigma_tnu/2. \
            -np.dot((nu_n-m*tau_n).T,((ttau/tau_n*(nu_n-m*tau_n)-2.*tnu)/(ttau+tau_n)))/2. \
            + (tnu**2/(tau_n+ttau)).sum()/2. - (np.log(1+ttau/tau_n)).sum()/2.
        return nlZ,nu_n,tau_n
    
    def epfitcRefresh(self,d0,P0,R0,R0P0,w,b):
        # refresh the representation of the posterior from initial and site parameters
        # to prevent possible loss of numerical precision after many epfitcUpdates
        # effort is O(n*nu^2) provided that nu<n
        # Sigma = inv(inv(K)+diag(W)) = diag(d) + P'*R0'*R'*R*R0*P.
        nu = R0.shape[0]                                 # number of inducing points
        rot180   = lambda A: np.rot90(np.rot90(A))       # little helper functions
        chol_inv = lambda A: np.linalg.solve( rot180( np.linalg.cholesky(rot180(A)) ),np.eye(nu)) # chol(inv(A))
        t  = 1/(1+d0*w)                                  # temporary variable O(n)
        d  = d0*t                                        # O(n)
        P  = np.tile(t.T,(nu,1))*P0                      # O(n*nu)
        T  = np.tile((w*t).T,(nu,1))*R0P0                # temporary variable O(n*nu^2)
        R  = chol_inv(np.eye(nu)+np.dot(R0P0,T.T))       # O(n*nu^3)
        nn = d*b                                                  # O(n)
        gg = np.dot(R0.T,np.dot(R.T,np.dot(R,np.dot(R0P0,t*b))))  # O(n*nu)
        return d,P,R,nn,gg
    
    def epfitcUpdate(self,d,P_i,R,nn,gg,w,b,ii,w_i,b_i,m,d0,P0,R0):
        dwi = w_i-w[ii]
        dbi = b_i-b[ii]
        hi = nn[ii] + m[ii] + np.dot(P_i.T,gg)           # posterior mean of site i O(nu)
        t = 1+dwi*d[ii]
        d[ii] = d[ii]/t                                  # O(1)
        nn[ii] = d[ii]*b_i                               # O(1)
        r = 1+d0[ii]*w[ii]
        r = (r*r)/dwi + r*d0[ii]
        v = np.dot(R,np.dot(R0,P0[:,ii]))
        v = np.reshape(v,(v.shape[0],1))
        r = 1/(r+np.dot(v.T,v))
        if r>0:
            R = cholupdate(R,np.sqrt(r)*np.dot(R.T,v),'-')
        else:
            R = cholupdate(R,np.sqrt(-r)*np.dot(R.T,v),'+')
        ttemp = np.dot(R0.T,np.dot(R.T,np.dot(R,np.dot(R0,P_i))))
        gg = gg + ((dbi-dwi*(hi-m[ii]))/t) * np.reshape(ttemp,(ttemp.shape[0],1)) # O(nu^2)
        w[ii] = w_i; b[ii] = b_i;                          # update site parameters O(1)
        P_i = P_i/t                                        # O(nu)
        return d,P_i,R,nn,gg,w,b
    
    def mvmZ(self,x,RVdd,t):
        # matrix vector multiplication with Z=inv(K+inv(W))
        Zx = t*x - np.dot(RVdd.T,np.dot(RVdd,x))
        return Zx

    def mvmK(self,al,V,d0):
        # matrix vector multiplication with approximate covariance matrix
        Kal = np.dot(V.T,np.dot(V,al)) + d0*al
        return Kal

    def Psi_lineFITC(self,s,dalpha,alpha,V,d0,m,likfunc,y,inffunc):
        # criterion Psi at alpha + s*dalpha for line search
        alpha = alpha + s*dalpha
        f = self.mvmK(alpha,V,d0) + m
        vargout = likfunc.proceed(y,f,None,inffunc,None,3) 
        lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2] 
        W = -d2lp
        Psi = np.dot(alpha.T,(f-m))/2. - lp.sum()
        return Psi[0],alpha,f,dlp,W

    def fitcRefresh(self,d0,P0,R0,R0P0, w):
        # refresh the representation of the posterior from initial and site parameters
        # to prevent possible loss of numerical precision after many epfitcUpdates
        # effort is O(n*nu^2) provided that nu<n
        # Sigma = inv(inv(K)+diag(W)) = diag(d) + P'*R0'*R'*R*R0*P.
        nu = R0.shape[0]                                  # number of inducing points
        rot180   = lambda A: np.rot90(np.rot90(A))        # little helper functions
        chol_inv = lambda A: np.linalg.solve( rot180( np.linalg.cholesky(rot180(A)) ),np.eye(nu)) # chol(inv(A))
        t  = 1/(1+d0*w)                                   # temporary variable O(n)
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
    def proceed(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        if not isinstance(likfunc, lik.Gauss):
            raise Exception ('Exact inference only possible with Gaussian likelihood')
        n, D = x.shape
        K = covfunc.proceed(x)                                 # evaluate covariance matrix
        m = meanfunc.proceed(x)                                # evaluate mean vector
        
        sn2   = np.exp(2*likfunc.hyp[0])                       # noise variance of likGauss
        L     = np.linalg.cholesky(K/sn2+np.eye(n)).T          # Cholesky factor of covariance with noise
        alpha = solve_chol(L,y-m)/sn2
        post = postStruct()
        post.alpha = alpha                                     # return the posterior parameters
        post.sW    = np.ones((n,1))/np.sqrt(sn2)               # sqrt of noise precision vector
        post.L     = L                                         # L = chol(eye(n)+sW*sW'.*K)

        if nargout>1:                                          # do we want the marginal likelihood?
            nlZ = np.dot((y-m).T,alpha)/2. + np.log(np.diag(L)).sum() + n*np.log(2*np.pi*sn2)/2. # -log marg lik
            if nargout>2:                                      # do we want derivatives?
                dnlZ = dnlZStruct(meanfunc, covfunc, likfunc)  # allocate space for derivatives
                Q = solve_chol(L,np.eye(n))/sn2 - np.dot(alpha,alpha.T) # precompute for convenience
                dnlZ.lik = [sn2*np.trace(Q)]
                if covfunc.hyp:
                    for ii in range(len(covfunc.hyp)):
                        dnlZ.cov[ii] = (Q*covfunc.proceed(x, None, ii)).sum()/2.
                if meanfunc.hyp:
                    for ii in range(len(meanfunc.hyp)): 
                        dnlZ.mean[ii] = np.dot(-meanfunc.proceed(x, ii).T,alpha)
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
    def proceed(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        if not isinstance(likfunc, lik.Gauss):               # NOTE: no explicit call to likGauss
            raise Exception ('Exact inference only possible with Gaussian likelihood')
        if not isinstance(covfunc, cov.FITCOfKernel):
            raise Exception('Only covFITC supported.')          # check cov

        diagK,Kuu,Ku = covfunc.proceed(x)                       # evaluate covariance matrix
        m  = meanfunc.proceed(x)                                # evaluate mean vector
        n, D = x.shape
        nu = Kuu.shape[0]

        sn2   = np.exp(2*likfunc.hyp[0])                        # noise variance of likGauss
        snu2  = 1.e-6*sn2                                       # hard coded inducing inputs noise
        Luu   = np.linalg.cholesky(Kuu+snu2*np.eye(nu)).T       # Kuu + snu2*I = Luu'*Luu
        V     = np.linalg.solve(Luu.T,Ku)                       # V = inv(Luu')*Ku => V'*V = Q
        g_sn2 = diagK + sn2 - np.array([(V*V).sum(axis=0)]).T   # g + sn2 = diag(K) + sn2 - diag(Q)
        Lu    = np.linalg.cholesky(np.eye(nu) + np.dot(V/np.tile(g_sn2.T,(nu,1)),V.T)).T  # Lu'*Lu=I+V*diag(1/g_sn2)*V'
        r     = (y-m)/np.sqrt(g_sn2)
        be    = np.linalg.solve(Lu.T,np.dot(V,r/np.sqrt(g_sn2)))
        iKuu  = solve_chol(Luu,np.eye(nu))                      # inv(Kuu + snu2*I) = iKuu
        
        post = postStruct()
        post.alpha = np.linalg.solve(Luu,np.linalg.solve(Lu,be))# return the posterior parameters
        post.L  = solve_chol(np.dot(Lu,Luu),np.eye(nu)) - iKuu  # Sigma-inv(Kuu)
        post.sW = np.ones((n,1))/np.sqrt(sn2)                   # unused for FITC prediction  with gp.m

        if nargout>1:                                           # do we want the marginal likelihood
            nlZ = np.log(np.diag(Lu)).sum() + (np.log(g_sn2).sum() + n*np.log(2*np.pi) + np.dot(r.T,r) - np.dot(be.T,be))/2. 
            if nargout>2:                                       # do we want derivatives?
                dnlZ = dnlZStruct(meanfunc, covfunc, likfunc)   # allocate space for derivatives
                al = r/np.sqrt(g_sn2) - np.dot(V.T,np.linalg.solve(Lu,be))/g_sn2 # al = (Kt+sn2*eye(n))\y
                B = np.dot(iKuu,Ku)
                w = np.dot(B,al)
                W = np.linalg.solve(Lu.T,V/np.tile(g_sn2.T,(nu,1)))
                for ii in range(len(covfunc.hyp)):
                    [ddiagKi,dKuui,dKui] = covfunc.proceed(x, None, ii)    # eval cov deriv
                    R = 2.*dKui-np.dot(dKuui,B)
                    v = ddiagKi - np.array([(R*B).sum(axis=0)]).T          # diag part of cov deriv
                    dnlZ.cov[ii] = ( np.dot(ddiagKi.T,1./g_sn2) + np.dot(w.T,(np.dot(dKuui,w)-2.*np.dot(dKui,al))) \
                                   - np.dot(al.T,(v*al)) - np.dot(np.array([(W*W).sum(axis=0)]),v) - (np.dot(R,W.T)*np.dot(B,W.T)).sum() )/2.
                    dnlZ.cov[ii] = dnlZ.cov[ii][0,0]
                dnlZ.lik = sn2*((1./g_sn2).sum() - (np.array([(W*W).sum(axis=0)])).sum() - np.dot(al.T,al))
                dKuui = 2*snu2
                R = -dKuui*B
                v = -np.array([(R*B).sum(axis=0)]).T     # diag part of cov deriv
                dnlZ.lik += (np.dot(w.T,np.dot(dKuui,w)) -np.dot(al.T,(v*al)) \
                                 - np.dot(np.array([(W*W).sum(axis=0)]),v) - (np.dot(R,W.T)*np.dot(B,W.T)).sum() )/2. 
                dnlZ.lik = list(dnlZ.lik[0])
                for ii in range(len(meanfunc.hyp)):
                    dnlZ.mean[ii] = np.dot(-meanfunc.proceed(x, ii).T, al) 
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

    def proceed(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        tol = 1e-6                           # tolerance for when to stop the Newton iterations
        smax = 2; Nline = 20; thr = 1e-4     # line search parameters
        maxit = 20                           # max number of Newton steps in f
        inffunc = self
        K = covfunc.proceed(x)               # evaluate the covariance matrix
        m = meanfunc.proceed(x)              # evaluate the mean vector
        n, D = x.shape
        Psi_old = np.inf                     # make sure while loop starts by the largest old objective val
        if self.last_alpha == None:          # find a good starting point for alpha and f
            alpha = np.zeros((n,1))
            f = np.dot(K,alpha) + m          # start at mean if sizes not match 
            vargout = likfunc.proceed(y, f, None, inffunc, None, 3)
            lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]
            W= -d2lp
            Psi_new = -lp.sum()
        else:
            alpha = self.last_alpha
            f = np.dot(K,alpha) + m                       # try last one
            vargout = likfunc.proceed(y, f, None, inffunc, None, 3)
            lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]
            W= -d2lp
            Psi_new = np.dot(alpha.T,(f-m))/2. - lp.sum() # objective for last alpha
            vargout = - likfunc.proceed(y, m, None, inffunc, None, 1)
            Psi_def =  vargout[0]                         # objective for default init f==m
            if Psi_def < Psi_new:                         # if default is better, we use it
                alpha = np.zeros((n,1))
                f = np.dot(K,alpha) + m 
                vargout = likfunc.proceed(y, f, None, inffunc, None, 3)
                lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]
                W=-d2lp; Psi_new = -lp.sum()
        isWneg = np.any(W<0)       # flag indicating whether we found negative values of W
        it = 0                     # this happens for the Student's t likelihood

        while (Psi_old - Psi_new > tol) and it<maxit:      # begin Newton
            Psi_old = Psi_new; it += 1
            if isWneg:                   # stabilise the Newton direction in case W has negative values
                W = np.maximum(W,0)      # stabilise the Hessian to guarantee postive definiteness
                tol = 1e-10              # increase accuracy to also get the derivatives right
            sW = np.sqrt(W); L = np.linalg.cholesky(np.eye(n) + np.dot(sW,sW.T)*K).T
            b = W*(f-m) + dlp; 
            dalpha = b - sW*solve_chol(L,sW*np.dot(K,b)) - alpha
            vargout = brentmin(0,smax,Nline,thr,self.Psi_line,4,dalpha,alpha,K,m,likfunc,y,inffunc)
            s = vargout[0]
            Psi_new = vargout[1]
            Nfun = vargout[2]
            alpha = vargout[3]
            f = vargout[4]
            dlp = vargout[5]
            W = vargout[6]
            isWneg = np.any(W<0)
        self.last_alpha = alpha                 # remember for next call
        vargout = likfunc.proceed(y,f,None,inffunc,None,4) 
        lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]; d3lp = vargout[3] 
        W = -d2lp; isWneg = np.any(W<0)
        post = postStruct()
        post.alpha = alpha                      # return the posterior parameters
        post.sW = np.sqrt(np.abs(W))*np.sign(W) # preserve sign in case of negative
        if isWneg:
            [ldA,iA,post.L] = self.logdetA(K,W,3)
            nlZ = np.dot(alpha.T,(f-m))/2. - lp.sum() + ldA/2.
            nlZ = nlZ[0] 
        else:
            sW = post.sW
            post.L = np.linalg.cholesky(np.eye(n)+np.dot(sW,sW.T)*K).T 
            nlZ = np.dot(alpha.T,(f-m))/2. + (np.log(np.diag(post.L))-np.reshape(lp,(lp.shape[0],))).sum()
            nlZ = nlZ[0]
        if nargout>2:                                           # do we want derivatives?
            dnlZ = dnlZStruct(meanfunc, covfunc, likfunc)       # allocate space for derivatives
            if isWneg:                                          # switch between Cholesky and LU decomposition mode
                Z = -post.L                                     # inv(K+inv(W))
                g = np.atleast_2d((iA*K).sum(axis=1)).T /2      # deriv. of ln|B| wrt W; g = diag(inv(inv(K)+diag(W)))/2
            else:
                Z = np.tile(sW,(1,n))*solve_chol(post.L,np.diag(np.reshape(sW,(sW.shape[0],)))) #sW*inv(B)*sW=inv(K+inv(W))
                C = np.linalg.solve(post.L.T,np.tile(sW,(1,n))*K)            # deriv. of ln|B| wrt W
                g = np.atleast_2d((np.diag(K)-(C**2).sum(axis=0).T)).T /2.   # g = diag(inv(inv(K)+W))/2
            dfhat = g* d3lp                 # deriv. of nlZ wrt. fhat: dfhat=diag(inv(inv(K)+W)).*d3lp/2
            for ii in range(len(covfunc.hyp)):                               # covariance hypers
                dK = covfunc.proceed(x, None, ii)
                dnlZ.cov[ii] = (Z*dK).sum()/2. - np.dot(alpha.T,np.dot(dK,alpha))/2.   # explicit part
                b = np.dot(dK,dlp)                                           # b-K*(Z*b) = inv(eye(n)+K*diag(W))*b
                dnlZ.cov[ii] -= np.dot(dfhat.T,b-np.dot(K,np.dot(Z,b)))      # implicit part
                dnlZ.cov[ii] = dnlZ.cov[ii][0,0]
            for ii in range(len(likfunc.hyp)):                               # likelihood hypers
                [lp_dhyp,dlp_dhyp,d2lp_dhyp] = likfunc.proceed(y,f,None,inffunc,ii,3)
                dnlZ.lik[ii] = -np.dot(g.T,d2lp_dhyp) - lp_dhyp.sum()        # explicit part
                b = np.dot(K,dlp_dhyp)                                       # b-K*(Z*b) = inv(eye(n)+K*diag(W))*b
                dnlZ.lik[ii] -= np.dot(dfhat.T,b-np.dot(K,np.dot(Z,b)))      # implicit part
                dnlZ.lik[ii] = dnlZ.lik[ii][0,0]
            for ii in range(len(meanfunc.hyp)):                              # mean hypers
                dm = meanfunc.proceed(x, ii)
                dnlZ.mean[ii] = -np.dot(alpha.T,dm)                          # explicit part
                dnlZ.mean[ii] -= np.dot(dfhat.T,dm-np.dot(K,np.dot(Z,dm)))   # implicit part
                dnlZ.mean[ii] = dnlZ.mean[ii][0,0]
            return post,nlZ[0],dnlZ
        else:
            return post, nlZ[0]



class FITC_Laplace(Inference):
    '''
    FITC-Laplace approximation to the posterior Gaussian process. The function is
    equivalent to infLaplace with the covariance function:
    Kt = Q + G; G = diag(g); g = diag(K-Q);  Q = Ku' * inv(Kuu + snu2 * eye(nu)) * Ku;
    where Ku and Kuu are covariances w.r.t. to inducing inputs xu and
    snu2 = sn2/1e6 is the noise of the inducing inputs. We fixed the standard
    deviation of the inducing inputs snu to be a one per mil of the measurement 
    noise's standard deviation sn. In case of a likelihood without noise
    parameter sn2, we simply use snu2 = 1e-6.
    '''
    def __init__(self):
        self.last_alpha = None

    def proceed(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        if not isinstance(covfunc, cov.FITCOfKernel):
            raise Exception('Only covFITC supported.')            
        tol = 1e-6                             # tolerance for when to stop the Newton iterations
        smax = 2; Nline = 100; thr = 1e-4      # line search parameters
        maxit = 20                             # max number of Newton steps in f
        inffunc = Laplace()
        diagK,Kuu,Ku = covfunc.proceed(x)      # evaluate the covariance matrix
        m = meanfunc.proceed(x)                # evaluate the mean vector
        if likfunc.hyp:                        # hard coded inducing inputs noise
            sn2  = np.exp(2.*likfunc.hyp[-1]) 
            snu2 = 1.e-6*sn2                   # similar to infFITC
        else:
            snu2 = 1.e-6        
        
        n, D = x.shape
        nu = Kuu.shape[0]
        rot180   = lambda A: np.rot90(np.rot90(A))      # little helper functions
        chol_inv = lambda A: np.linalg.solve( rot180( np.linalg.cholesky(rot180(A)) ),np.eye(nu)) # chol(inv(A))
        R0 = chol_inv(Kuu+snu2*np.eye(nu))              # initial R, used for refresh O(nu^3)
        V  = np.dot(R0,Ku); d0 = diagK - np.array([(V*V).sum(axis=0)]).T     # initial d, needed
    
        Psi_old = np.inf                    # make sure while loop starts by the largest old objective val
        if self.last_alpha == None:         # find a good starting point for alpha and f
            alpha = np.zeros((n,1))
            f = self.mvmK(alpha,V,d0) + m   # start at mean if sizes not match 
            vargout = likfunc.proceed(y, f, None, inffunc, None, 3)
            lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]
            W=-d2lp; Psi_new = -lp.sum()
        else:
            alpha = self.last_alpha
            f = self.mvmK(alpha,V,d0) + m                           # try last one
            vargout = likfunc.proceed(y, f, None, inffunc, None, 3)
            lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]
            W=-d2lp
            Psi_new = np.dot(alpha.T,(f-m))/2. - lp.sum()           # objective for last alpha
            vargout = - likfunc.proceed(y, m, None, inffunc, None, 1)
            Psi_def =  vargout[0]                                   # objective for default init f==m
            if Psi_def < Psi_new:                                   # if default is better, we use it
                alpha = np.zeros((n,1))
                f = self.mvmK(alpha,V,d0) + m
                vargout = likfunc.proceed(y, f, None, inffunc, None, 3)
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
            b = W*(f-m) + dlp; dd = 1/(1+W*d0)
            RV = np.dot( chol_inv( np.eye(nu) + np.dot(V*np.tile((W*dd).T,(nu,1)),V.T)),V ) 
            dalpha = dd*b - (W*dd)*np.dot(RV.T,np.dot(RV,(dd*b))) - alpha # Newt dir + line search
            vargout = brentmin(0,smax,Nline,thr,self.Psi_lineFITC,4,dalpha,alpha,V,d0,m,likfunc,y,inffunc)
            s = vargout[0]; Psi_new = vargout[1]; Nfun = vargout[2]; alpha = vargout[3]
            f = vargout[4]; dlp = vargout[5]; W = vargout[6]
            isWneg = np.any(W<0)

        self.last_alpha = alpha                                     # remember for next call
        vargout = likfunc.proceed(y,f,None,inffunc,None,4) 
        lp = vargout[0]; dlp = vargout[1]; d2lp = vargout[2]; d3lp = vargout[3]  

        W=-d2lp; isWneg = np.any(W<0)
        post = postStruct()
        post.alpha = np.dot(R0.T,np.dot(V,alpha))                   # return the posterior parameters
        post.sW = np.sqrt(np.abs(W))*np.sign(W)                     # preserve sign in case of negative
        dd = 1/(1+d0*W)                                             # temporary variable O(n)
        A = np.eye(nu) + np.dot(V*np.tile((W*dd).T,(nu,1)),V.T)     # temporary variable O(n*nu^2)
        R0tV = np.dot(R0.T,V); B = R0tV*np.tile((W*dd).T,(nu,1))    # temporary variables O(n*nu^2)
        post.L = -np.dot(B,R0tV.T)                                  # L = -R0'*V*inv(Kt+diag(1./ttau))*V'*R0, first part
        if np.any(1+d0*W<0):
            raise Exception('W is too negative; nlZ and dnlZ cannot be computed.')
        nlZ = np.dot(alpha.T,(f-m))/2. - lp.sum() - np.log(dd).sum()/2. + \
            np.log(np.diag(np.linalg.cholesky(A).T)).sum()
        RV = np.dot(chol_inv(A),V)
        RVdd = RV * np.tile((W*dd).T,(nu,1))                        # RVdd needed for dnlZ
        B = np.dot(B,RV.T)
        post.L += np.dot(B,B.T)

        if nargout>2:                                               # do we want derivatives?
            dnlZ = dnlZStruct(meanfunc, covfunc, likfunc)           # allocate space for derivatives
            [d,P,R] = self.fitcRefresh(d0,Ku,R0,V,W)                # g = diag(inv(inv(K)+W))/2
            g = d/2 + 0.5*np.atleast_2d((np.dot(np.dot(R,R0),P)**2).sum(axis=0)).T
            t = W/(1+W*d0)
            
            dfhat = g*d3lp  # deriv. of nlZ wrt. fhat: dfhat=diag(inv(inv(K)+W)).*d3lp/2
            for ii in range(len(covfunc.hyp)):                      # covariance hypers
                ddiagK,dKuu,dKu = covfunc.proceed(x, None, ii)      # eval cov derivatives
                dA = 2.*dKu.T-np.dot(R0tV.T,dKuu)                   # dQ = dA*R0tV
                w = np.atleast_2d((dA*R0tV.T).sum(axis=1)).T        # w = diag(dQ)
                v = ddiagK-w                                        # v = diag(dK)-diag(dQ);
                dnlZ.cov[ii] = np.dot(ddiagK.T,t) - np.dot((RVdd*RVdd).sum(axis=0),v)   # explicit part
                dnlZ.cov[ii] -= (np.dot(RVdd,dA)*np.dot(RVdd,R0tV.T)).sum()             # explicit part
                dnlZ.cov[ii] = 0.5*dnlZ.cov[ii] - np.dot(alpha.T,np.dot(dA,np.dot(R0tV,alpha))+v*alpha)/2. # explicit
                b = np.dot(dA,np.dot(R0tV,dlp)) + v*dlp             # b-K*(Z*b) = inv(eye(n)+K*diag(W))*b
                KZb = self.mvmK(self.mvmZ(b,RVdd,t),V,d0)
                dnlZ.cov[ii] -= np.dot(dfhat.T,(b-KZb))             # implicit part
                dnlZ.cov[ii] = dnlZ.cov[ii][0,0]
                
            for ii in range(len(likfunc.hyp)):                      # likelihood hypers
                vargout = likfunc.proceed(y,f,None,inffunc,ii,3)
                lp_dhyp = vargout[0]; dlp_dhyp = vargout[1]; d2lp_dhyp = vargout[2] 
                dnlZ.lik[ii] = -np.dot(g.T,d2lp_dhyp) - lp_dhyp.sum() # explicit part
                b = self.mvmK(dlp_dhyp,V,d0)                          # implicit part
                dnlZ.lik[ii] -= np.dot(dfhat.T,b-self.mvmK(self.mvmZ(b,RVdd,t),V,d0))
                if ii == len(likfunc.hyp)-1:
                    # since snu2 is a fixed fraction of sn2, there is a covariance-like term
                    # in the derivative as well
                    snu = np.sqrt(snu2);
                    T = chol_inv(Kuu + snu2*np.eye(nu)); 
                    T = np.dot(T.T,np.dot(T,snu*Ku)); 
                    t = np.array([(T*T).sum(axis=0)]).T 
                    z = np.dot(alpha.T,np.dot(T.T,np.dot(T,alpha))-t*alpha) - np.dot(np.array([(RVdd*RVdd).sum(axis=0)]),t)
                    z += (np.dot(RVdd,T.T)**2).sum()
                    b = (t*dlp-np.dot(T.T,np.dot(T,dlp)))/2.
                    KZb = self.mvmK(self.mvmZ(b,RVdd,t),V,d0)
                    z -= np.dot(dfhat.T,b-KZb)
                    dnlZ.lik[ii] += z
                    dnlZ.lik[ii] = dnlZ.lik[ii][0,0]

            for ii in range(len(meanfunc.hyp)):                           # mean hypers
                dm = meanfunc.proceed(x, ii)
                dnlZ.mean[ii] = -np.dot(alpha.T,dm)                       # explicit part
                Zdm = self.mvmZ(dm,RVdd,t)
                dnlZ.mean[ii] -= np.dot(dfhat.T,(dm-self.mvmK(Zdm,V,d0))) # implicit part
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
    def proceed(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        tol = 1e-4; max_sweep = 10; min_sweep = 2 # tolerance to stop EP iterations
        n = x.shape[0]
        inffunc = self
        K = covfunc.proceed(x)                    # evaluate the covariance matrix
        m = meanfunc.proceed(x)                   # evaluate the mean vector
        nlZ0 = -likfunc.proceed(y, m, np.reshape(np.diag(K),(np.diag(K).shape[0],1)), inffunc).sum()
        if self.last_ttau == None:                # find starting point for tilde parameters
            ttau  = np.zeros((n,1))               # initialize to zero if we have no better guess
            tnu   = np.zeros((n,1))
            Sigma = K                             # initialize Sigma and mu, the parameters of ..
            mu    = np.zeros((n,1))               # .. the Gaussian posterior approximation
            nlZ   = nlZ0
        else:
            ttau = self.last_ttau                 # try the tilde values from previous call
            tnu  = self.last_tnu
            Sigma, mu, nlZ, L = self.epComputeParams(K, y, ttau, tnu, likfunc, m, inffunc)
            if nlZ > nlZ0:                        # if zero is better ..
                ttau = np.zeros((n,1))            # .. then initialize with zero instead
                tnu  = np.zeros((n,1)) 
                Sigma = K                         # initialize Sigma and mu, the parameters of ..
                mu = np.zeros((n,1))              # .. the Gaussian posterior approximation
                nlZ = nlZ0
        nlZ_old = np.inf; sweep = 0               # converged, max. sweeps or min. sweeps?
        while (np.abs(nlZ-nlZ_old) > tol and sweep < max_sweep) or (sweep < min_sweep):
            nlZ_old = nlZ; sweep += 1
            rperm = xrange(n)                     # randperm(n)
            for ii in rperm:                      # iterate EP updates (in random order) over examples
                tau_ni = 1/Sigma[ii,ii] - ttau[ii]#  first find the cavity distribution ..
                nu_ni  = mu[ii]/Sigma[ii,ii] + m[ii]*tau_ni - tnu[ii]    # .. params tau_ni and nu_ni
                # compute the desired derivatives of the indivdual log partition function
                lZ,dlZ,d2lZ = likfunc.proceed(y[ii], nu_ni/tau_ni, 1/tau_ni, inffunc, None, 3)
                ttau_old = copy(ttau[ii])         # then find the new tilde parameters, keep copy of old
                ttau[ii] = -d2lZ  /(1.+d2lZ/tau_ni)
                ttau[ii] = max(ttau[ii],0)        # enforce positivity i.e. lower bound ttau by zero
                tnu[ii]  = ( dlZ + (m[ii]-nu_ni/tau_ni)*d2lZ )/(1.+d2lZ/tau_ni)
                ds2 = ttau[ii] - ttau_old         # finally rank-1 update Sigma ..
                si  = np.reshape(Sigma[:,ii],(Sigma.shape[0],1))
                Sigma = Sigma - ds2/(1.+ds2*si[ii])*np.dot(si,si.T)   # takes 70# of total time
                mu = np.dot(Sigma,tnu)                                # .. and recompute mu
            # recompute since repeated rank-one updates can destroy numerical precision
            Sigma, mu, nlZ, L = self.epComputeParams(K, y, ttau, tnu, likfunc, m, inffunc)
        if sweep == max_sweep:
            raise Exception('maximum number of sweeps reached in function infEP')
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
            tau_n = 1/Dsigma-ttau                           # compute the log marginal likelihood
            nu_n  = mu/Dsigma-tnu                           # vectors of cavity parameters
            F = np.dot(alpha,alpha.T) - np.tile(sW,(1,n))* \
                solve_chol(L,np.diag(np.reshape(sW,(sW.shape[0],))))   # covariance hypers
            for jj in range(len(covfunc.hyp)):
                dK = covfunc.proceed(x, None, jj)
                dnlZ.cov[jj] = -(F*dK).sum()/2.
            for ii in range(len(likfunc.hyp)):
                dlik = likfunc.proceed(y, nu_n/tau_n, 1/tau_n, inffunc, ii)
                dnlZ.lik[ii] = -dlik.sum()    
            junk,dlZ = likfunc.proceed(y, nu_n/tau_n, 1/tau_n, inffunc, None, 2) # mean hyps
            for ii in range(len(meanfunc.hyp)):
                dm = meanfunc.proceed(x, ii)
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

    def proceed(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        if not isinstance(covfunc, cov.FITCOfKernel):
            raise Exception('Only covFITC supported.')  # check cov
        tol = 1e-4; max_sweep = 10; min_sweep = 2       # tolerance to stop EP iterations
        inffunc = EP()

        diagK,Kuu,Ku = covfunc.proceed(x)               # evaluate the covariance matrix
        m = meanfunc.proceed(x)                         # evaluate the mean vector

        if likfunc.hyp:                                 # hard coded inducing inputs noise
            sn2  = np.exp(2.*likfunc.hyp[-1]) 
            snu2 = 1.e-6*sn2                            # similar to infFITC
        else:
            snu2 = 1.e-6
    
        n, D = x.shape; nu = Kuu.shape[0]
        rot180   = lambda A: np.rot90(np.rot90(A))      # little helper functions
        chol_inv = lambda A: np.linalg.solve( rot180( np.linalg.cholesky(rot180(A)) ),np.eye(nu)) # chol(inv(A))

        R0 = chol_inv(Kuu+snu2*np.eye(nu))              # initial R, used for refresh O(nu^3)
        V  = np.dot(R0,Ku); d0 = diagK - np.array([(V*V).sum(axis=0)]).T # initial d, needed for refresh O(n*nu^2)

        # A note on naming: variables are given short but descriptive names in 
        # accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
        # and s2 are mean and variance, nu and tau are natural parameters. A leading t
        # means tilde, a subscript _ni means "not i" (for cavity parameters), or _n
        # for a vector of cavity parameters.

        # marginal likelihood for ttau = tnu = zeros(n,1); equals n*log(2) for likCum*
        nlZ0 = -1.* likfunc.proceed(y, m, np.reshape(diagK,(diagK.shape[0],1)), inffunc).sum()
        if self.last_ttau == None:                      # find starting point for tilde parameters
            ttau  = np.zeros((n,1))                     # initialize to zero if we have no better guess
            tnu   = np.zeros((n,1))
            [d,P,R,nn,gg] = self.epfitcRefresh(d0,Ku,R0,V,ttau,tnu)   # compute initial repres.
            nlZ = nlZ0
        else:
            ttau = self.last_ttau                       # try the tilde values from previous call
            tnu  = self.last_tnu
            [d,P,R,nn,gg] = self.epfitcRefresh(d0,Ku,R0,V,ttau,tnu) # compute initial repres.
            nlZ = self.epfitcZ(d,P,R,nn,gg,ttau,tnu,d0,R0,Ku,y,likfunc,m,inffunc)[0]
            if nlZ > nlZ0:                              # if zero is better ..
                ttau = np.zeros((n,1))                  # .. then initialize with zero instead
                tnu  = np.zeros((n,1))
                [d,P,R,nn,gg] = self.epfitcRefresh(d0,Ku,R0,V,ttau,tnu) # initial repres.
                nlZ = nlZ0

        nlZ_old = np.inf; sweep = 0                     # converged, max. sweeps or min. sweeps?
        while (np.abs(nlZ-nlZ_old) > tol and sweep < max_sweep) or (sweep < min_sweep):
            nlZ_old = nlZ
            sweep += 1
            rperm = range(n)                            # randperm(n)
            for ii in rperm:                            # iterate EP updates (in random order) over examples
                p_i = np.reshape(P[:,ii],(P.shape[0],1))
                t = np.dot(R,np.dot(R0,p_i))            # temporary variables
                sigma_i = d[ii] + np.dot(t.T,t); mu_i = nn[ii] + np.dot(p_i.T,gg) # post moments O(nu^2)    
                tau_ni = 1/sigma_i - ttau[ii]                   #  first find the cavity distribution ..
                nu_ni  = mu_i/sigma_i + m[ii]*tau_ni - tnu[ii]  # .. params tau_ni and nu_ni
                # compute the desired derivatives of the indivdual log partition function
                vargout = likfunc.proceed(y[ii], nu_ni/tau_ni, 1/tau_ni, inffunc, None, 3)
                lZ = vargout[0]; dlZ = vargout[1]; d2lZ = vargout[2]
                ttau_i = -d2lZ  /(1.+d2lZ/tau_ni)
                ttau_i = max(ttau_i,0)                  # enforce positivity i.e. lower bound ttau by zero
                tnu_i  = ( dlZ + (m[ii]-nu_ni/tau_ni)*d2lZ )/(1.+d2lZ/tau_ni)
                [d,P[:,ii],R,nn,gg,ttau,tnu] = self.epfitcUpdate(d,P[:,ii],R,nn,gg,ttau,tnu,ii,ttau_i,tnu_i,m,d0,Ku,R0)# update representation
      
            # recompute since repeated rank-one updates can destroy numerical precision
            [d,P,R,nn,gg] = self.epfitcRefresh(d0,Ku,R0,V,ttau,tnu)
            [nlZ,nu_n,tau_n] = self.epfitcZ(d,P,R,nn,gg,ttau,tnu,d0,R0,Ku,y,likfunc,m,inffunc)
        if sweep == max_sweep:
            raise Exception('maximum number of sweeps reached in function infEP')
        
        self.last_ttau = ttau
        self.last_tnu = tnu       # remember for next call
        post = postStruct()
        post.sW = np.sqrt(ttau)   # unused for FITC_EP prediction with gp.m
        dd = 1/(d0+1/ttau)
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
                ddiagK,dKuu,dKu = covfunc.proceed(x, None, ii)
                dA = 2*dKu.T - np.dot(R0tV.T,dKuu)            # dQ = dA*R0tV
                w = np.atleast_2d((dA*R0tV.T).sum(axis=1)).T  # w = diag(dQ)
                v = ddiagK - w                                # v = diag(dK)-diag(dQ)
                z = np.dot(dd.T,(v+w)) - np.dot(np.atleast_2d((RVdd*RVdd).sum(axis=0)), v) \
                       - (np.dot(RVdd,dA).T * np.dot(R0tV,RVdd.T)).sum()
                dnlZ.cov[ii] = (z - np.dot(alpha.T,(alpha*v)) - np.dot(np.dot(alpha.T,dA),np.dot(R0tV,alpha)))/2.
                dnlZ.cov[ii] = dnlZ.cov[ii][0,0]
            for ii in range(len(likfunc.hyp)):                # likelihood hypers
                dlik = likfunc.proceed(y, nu_n/tau_n+m, 1/tau_n, inffunc, ii, 1)
                dnlZ.lik[ii] = -dlik.sum()                                 
                if ii == len(likfunc.hyp)-1:
                    # since snu2 is a fixed fraction of sn2, there is a covariance-like term
                    # in the derivative as well
                    v = np.atleast_2d((R0tV*R0tV).sum(axis=0)).T
                    z = (np.dot(RVdd,R0tV.T)**2).sum()  - np.dot(np.atleast_2d((RVdd*RVdd).sum(axis=0)),v)
                    z = z + np.dot(post.alpha.T,post.alpha) - np.dot(alpha.T,(v*alpha))
                    dnlZ.lik[ii] += snu2*z
                    dnlZ.lik[ii] = dnlZ.lik[ii][0,0]
            [junk,dlZ] = likfunc.proceed(y, nu_n/tau_n, 1/tau_n, inffunc, None, 2) # mean hyps
            for ii in range(len(meanfunc.hyp)):
                dm = meanfunc.proceed(x, ii)
                dnlZ.mean[ii] = -np.dot(dlZ.T,dm)
                dnlZ.mean[ii] = dnlZ.mean[ii][0,0]
        
            return post, nlZ[0,0], dnlZ
        else:
            return post, nlZ[0,0]
    





# test
if __name__ == '__main__':
    pass








		