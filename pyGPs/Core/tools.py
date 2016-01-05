from __future__ import division
from __future__ import print_function
from past.builtins import cmp
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

# This is a object-oriented python implementation of gpml functionality
# (Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18).
# based on the functional-version of python implementation
# (Copyright (c) by Marion Neumann and Daniel Marthaler, 20/05/2013)
#
# Copyright (c) by Marion Neumann and Shan Huang, 30/09/2013

import numpy as np
import matplotlib.pyplot as plt
import sys
from math import sqrt
import scipy.linalg.lapack as lapack

def jitchol(A,maxtries=5):
    ''' Copyright (c) 2012, GPy authors (James Hensman, Nicolo Fusi, Ricardo Andrade,
        Nicolas Durrande, Alan Saul, Max Zwiessele, Neil D. Lawrence).
    All rights reserved
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
      * Neither the name of the <organization> nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    :param A: the matrixed to be decomposited
    :param int maxtries: number of iterations of adding jitters
    '''    
    A = np.asfortranarray(A)
    L, info = lapack.dpotrf(A, lower=1)
    if info == 0:
        return L
    else:
        diagA = np.diag(A)
        if np.any(diagA <= 0.):
            raise np.linalg.LinAlgError("kernel matrix not positive definite: non-positive diagonal elements")
        jitter = diagA.mean() * 1e-9
        while maxtries > 0 and np.isfinite(jitter):
            print('Warning: adding jitter of {:.10e} to diagnol of kernel matrix for numerical stability'.format(jitter))
            try:
                return np.linalg.cholesky(A + np.eye(A.shape[0]).T * jitter, lower=True)
            except:
                jitter *= 10
            finally:
                maxtries -= 1
        raise np.linalg.LinAlgError("kernel matrix not positive definite, even with jitter.")



def solve_chol(L, B):
    '''
    Solve linear equations from the Cholesky factorization.
    Solve A*X = B for X, where A is square, symmetric, positive definite. The
    input to the function is L the Cholesky decomposition of A and the matrix B.
    Example: X = solve_chol(chol(A),B)

    :param L: low trigular matrix (cholesky decomposition of A)
    :param B: matrix have the same first dimension of L
    :return: X = A \ B
    '''
    try:
        assert(L.shape[0] == L.shape[1] and L.shape[0] == B.shape[0])
    except AssertionError:
        raise Exception('Wrong sizes of matrix arguments in solve_chol.py');
    X = np.linalg.solve(L,np.linalg.solve(L.T,B))
    return X



def unique(x):
    '''
    Return a list with unique elements.

    :param x: any matrix x
    :return: a list of unique elements in x
    '''
    # First flatten x
    y = [item for sublist in x for item in sublist]
    if isinstance(x,np.ndarray):
        n,D = x.shape
        assert(D == 1)
        y = np.array( list(set(x[:,0])) )
        y = np.reshape(y, (len(y),1))
    else:
        y = list(set(y))
    return y



def brentmin(xlow,xupp,Nitmax,tol,f,nout=None,*args):
    '''
    Brent's minimization method in one dimension. 
    Given a function f, and given a search interval this routine isolates 
    the minimum of fractional precision of about tol using Brent's method.
    Reference: Section 10.2 Parabolic Interpolation and Brent's Method in One Dimension
    Press, Teukolsky, Vetterling & Flannery
    Numerical Recipes in C, Cambridge University Press, 2002
    This is a python implementation of gpml functionality (Copyright (c) by
    Hannes Nickisch 2010-01-10). 
    xmin,fmin,funccout,varargout = BRENTMIN(xlow,xupp,Nit,tol,f,nout,varargin)

    :param xlow: lower bound. i.e. search interval such that xlow<=xmin<=xupp
    :param xupp: uppper bound. i.e. search interval such that xlow<=xmin<=xupp
    :param Nitmax: maximum number of function evaluations made by the routine
    :param tol: fractional precision 
    :param f:  [y,varargout{:}] = f(x,varargin{:}) is the function
    :param nout: no. of outputs of f (in varargout) in addition to the y value

    :return:  fmin is minimal function value. xmin is corresponding abscissa-value
    funccount is the number of function evaluations made. varargout is additional outputs of f at optimum.
    '''
    # code taken from
    #    Section 10.2 Parabolic Interpolation and Brent's Method in One Dimension
    #    Press, Teukolsky, Vetterling & Flannery
    #    Numerical Recipes in C, Cambridge University Press, 2002
    #
    # [xmin,fmin,funccout,varargout] = BRENTMIN(xlow,xupp,Nit,tol,f,nout,varargin)
    #    Given a function f, and given a search interval this routine isolates 
    #    the minimum of fractional precision of about tol using Brent's method.
    # 
    # INPUT
    # -----
    # xlow,xupp:  search interval such that xlow<=xmin<=xupp
    # Nitmax:     maximum number of function evaluations made by the routine
    # tol:        fractional precision 
    # f:          [y,varargout{:}] = f(x,varargin{:}) is the function
    # nout:       no. of outputs of f (in varargout) in addition to the y value
    #
    # OUTPUT
    # ------
    # fmin:      minimal function value
    # xmin:      corresponding abscissa-value
    # funccount: number of function evaluations made
    # varargout: additional outputs of f at optimum
    #
    # This is a python implementation of gpml functionality (Copyright (c) by
    # Hannes Nickisch 2010-01-10).
    

    if nout == None:
        nout = 0
    eps = sys.float_info.epsilon
    # tolerance is no smaller than machine's floating point precision
    tol = max(tol,eps)
    # Evaluate endpoints
    vargout = f(xlow,*args); fa = vargout[0][0]
    vargout = f(xupp,*args); fb = vargout[0][0]
    funccount = 2 # number of function evaluations
    # Compute the start point
    seps = sqrt(eps)
    c = 0.5*(3.0 - sqrt(5.0)) # golden ratio
    a = xlow
    b = xupp
    v = a + c*(b-a)
    w = v
    xf = v
    d = 0.
    e = 0.
    x = xf
    vargout = f(x,*args)
    fx = vargout[0][0]
    varargout = vargout[1:]
    funccount += 1
    fv = fx; fw = fx
    xm = 0.5*(a+b)
    tol1 = seps*abs(xf) + old_div(tol,3.0);
    tol2 = 2.0*tol1
    # Main loop
    while ( abs(xf-xm) > (tol2 - 0.5*(b-a)) ):
        gs = True
        # Is a parabolic fit possible
        if abs(e) > tol1:
            # Yes, so fit parabola
            gs = False
            r = (xf-w)*(fx-fv)
            q = (xf-v)*(fx-fw)
            p = (xf-v)*q-(xf-w)*r
            q = 2.0*(q-r)
            if q > 0.0:  
                p = -p
            q = abs(q)
            r = e;  e = d
            # Is the parabola acceptable
            if ( (abs(p)<abs(0.5*q*r)) and (p>q*(a-xf)) and (p<q*(b-xf)) ):
                # Yes, parabolic interpolation step
                d = old_div(p,q)
                x = xf+d
                # f must not be evaluated too close to ax or bx
                if ((x-a) < tol2) or ((b-x) < tol2):
                    si = cmp(xm-xf,0)
                    if ((xm-xf) == 0): si += 1
                    d = tol1*si
            else:
                # Not acceptable, must do a golden section step
                gs = True
        if gs:
            # A golden-section step is required
            if xf >= xm: e = a-xf    
            else: 
                e = b-xf
            d = c*e
        # The function must not be evaluated too close to xf
        si = cmp(d,0)
        if (d == 0): si += 1
        x = xf + si * max(abs(d),tol1)
        vargout = f(x,*args); fu = vargout[0][0]; varargout = vargout[1:]
        funccount += 1
        # Update a, b, v, w, x, xm, tol1, tol2
        if fu <= fx:
            if x >= xf: a = xf 
            else: b = xf
            v = w; fv = fw
            w = xf; fw = fx
            xf = x; fx = fu
        else: # fu > fx
            if x < xf: 
                a = x
            else: 
                b = x 
            if ( (fu <= fw) or (w == xf) ):
                v = w; fv = fw
                w = x; fw = fu
            elif ( (fu <= fv) or ((v == xf) or (v == w)) ):
                v = x; fv = fu
        xm = 0.5*(a+b)
        tol1 = seps*abs(xf) + old_div(tol,3.0); tol2 = 2.0*tol1
        if funccount >= Nitmax:
            # typically we should not get here
            # print 'Warning: Specified number of function evaluation reached (brentmin)'
            break
    # check that endpoints are less than the minimum found
    if ( (fa < fx) and (fa <= fb) ):
        xf = xlow; fx = fa
    elif fb < fx:
        xf = xupp; fx = fb
    fmin = fx
    xmin = xf
    vargout = [xmin,fmin,funccount]
    for vv in varargout:
        vargout.append(vv)
    return vargout



def cholupdate(R,x,sgn='+'):
    '''
    Placeholder for a python version of MATLAB's cholupdate.  Now it is O(n^3)
    '''
    if len(x.shape) == 1:
        # Reshape x so that the dot product below is correct
        x = np.reshape(x,(x.shape[0],1))
    assert(R.shape[0] == x.shape[0])
    A = np.dot(R.T,R)
    if sgn == '+':
        R1 = A + np.dot(x,x.T)
    elif sgn == '-':
        R1 = A - np.dot(x,x.T)
    else:
        raise Exception('Sign needs to be + or - in cholupdate')
    return jitchol(R1).T



