#!/usr/bin/python
# -*- coding: utf-8 -*-

# ========================================================================
#   This program is distributed WITHOUT ANY WARRANTY; without even the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#
#   This file contains a Python version of Carl Rasmussen's Matlab-function
#   minimize.m
#
#   minimize.m is copyright (C) 1999 - 2006, Carl Edward Rasmussen.
#   Python adaptation by Roland Memisevic 2008.
#   updates by Shan Huang 2013
#
#
#   The following is the original copyright notice that comes with the
#   function minimize.m
#   (from http://www.kyb.tuebingen.mpg.de/bs/people/carl/code/minimize/Copyright):
#   Rasmussen
#
#   "(C) Copyright 1999 - 2006, Carl Edward Rasmussen
#
#   Permission is granted for anyone to copy, use, or modify these
#   programs and accompanying documents for purposes of research or
#   education, provided this copyright notice is retained, and note is
#   made of any changes that have been made.
#
#   These programs and documents are distributed without any warranty,
#   express or implied.  As the programs were written for research
#   purposes only, they have not been tested to the degree that would be
#   advisable in any important application.  All use of these programs is
#   entirely at the user's own risk."
# ========================================================================

"""

This module contains a function  that performs unconstrained
gradient based optimization using nonlinear conjugate gradients.

The function is a straightforward Python-translation of Carl Rasmussen's
Matlab-function minimize.m

"""

from numpy import dot, isinf, isnan, any, sqrt, isreal, real, nan, inf


def rt_minimize(X,f,length=-100,*args):
    red = 1.0
    verbose = False
    # don't reevaluate within 0.1 of the limit of the current bracket
    INT = 0.1
    EXT = 3.0  # extrapolate maximum 3 times the current step-size
    MAX = 20  # max 20 function evaluations per line search
    RATIO = 10  # maximum allowed slope ratio
    SIG = 0.1  # SIG and RHO are the constants controlling the Wolfe-
    RHO = SIG / 2

    # Powell conditions. SIG is the maximum allowed absolute ratio between
    # previous and new slopes (derivatives in the search direction), thus setting
    # SIG to low (positive) values forces higher precision in the line-searches.
    # RHO is the minimum allowed fraction of the expected (from the slope at the
    # initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
    # Tuning of SIG (depending on the nature of the function to be optimized) may
    # speed up the minimization; it is probably not worth playing much with
    # RHO.

    SMALL = 10. ** -16  # minimize.m uses matlab's realmin
    i = 0  # zero the run length counter
    ls_failed = 0  # no previous line search has failed
    result = f(X, *args)
    f0 = result[0]  # get function value and gradient
    df0 = result[1]
    fX = [f0]
    i = i + (length < 0)  # count epochs?!
    s = -df0
    d0 = -dot(s, s)  # initial search direction (steepest) and slope
    x3 = red / (1.0 - d0)  # initial step is red/(|s|+1)

    while i < abs(length):  # while not finished
        i = i + (length > 0)  # count iterations?!

        X0 = X  # make a copy of current values
        F0 = f0
        dF0 = df0
        if length > 0:
            M = MAX
        else:
            M = min(MAX, -length - i)
        while 1:  # keep extrapolating as long as necessary
            x2 = 0
            f2 = f0
            d2 = d0
            f3 = f0
            df3 = df0
            success = 0
            while not success and M > 0:
                try:
                    M = M - 1  # count epochs?!
                    i = i + (length < 0)
                    result3 = f(X + x3 * s, *args)
                    f3 = result3[0]
                    df3 = result3[1]
                    if isnan(f3) or isinf(f3) or any(isnan(df3) + isinf(df3)):
                        return
                    success = 1
                except:
					# catch any error which occured in f
                    x3 = (x2 + x3) / 2  # bisect and try again
            if f3 < F0:
                X0 = X + x3 * s  # keep best values
                F0 = f3
                dF0 = df3
            d3 = dot(df3, s)  # new slope
            if d3 > SIG * d0 or f3 > f0 + x3 * RHO * d0 or M == 0:
				# are we done extrapolating?
                break
            x1 = x2  # move point 2 to point 1
            f1 = f2
            d1 = d2
            x2 = x3  # move point 3 to point 2
            f2 = f3
            d2 = d3
            # make cubic extrapolation
            A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
            B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
            Z = B + sqrt(complex(B * B - A * d1 * (x2 - x1)))
            if Z != 0.0:
                x3 = x1 - d1 * (x2 - x1) ** 2 / Z  # num. error possible, ok!
            else:
                x3 = inf
            if not isreal(x3) or isnan(x3) or isinf(x3) or x3 < 0:
				# num prob | wrong sign?
                x3 = x2 * EXT  # extrapolate maximum amount
            elif x3 > x2 * EXT:
				# new point beyond extrapolation limit?
                x3 = x2 * EXT  # extrapolate maximum amount
            elif x3 < x2 + INT * (x2 - x1):
				# new point too close to previous point?
                x3 = x2 + INT * (x2 - x1)
            x3 = real(x3)

        while (abs(d3) > -SIG * d0 or f3 > f0 + x3 * RHO * d0) and M > 0:
			# keep interpolating
            if d3 > 0 or f3 > f0 + x3 * RHO * d0:  # choose subinterval
                x4 = x3  # move point 3 to point 4
                f4 = f3
                d4 = d3
            else:
                x2 = x3  # move point 3 to point 2
                f2 = f3
                d2 = d3
            if f4 > f0:
                x3 = x2 - 0.5 * d2 * (x4 - x2) ** 2 / (f4 - f2 - d2 * (x4 - x2))
            else:
				# quadratic interpolation
                # cubic interpolation
                A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)
                B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                if A != 0:
                    x3 = x2 + (sqrt(B * B - A * d2 * (x4 - x2) ** 2) - B) / A
                else:
					# num. error possible, ok!
                    x3 = inf
            if isnan(x3) or isinf(x3):
                x3 = (x2 + x4) / 2  # if we had a numerical problem then bisect
            x3 = max(min(x3, x4 - INT * (x4 - x2)), x2 + INT * (x4 - x2))
				# dont accept too close
            result3 = f(X + x3 * s, *args)
            f3 = result3[0]
            df3 = result3[1]
            if f3 < F0:
                X0 = X + x3 * s  # keep best values
                F0 = f3
                dF0 = df3
            M = M - 1  # count epochs?!
            i = i + (length < 0)
            d3 = dot(df3, s)  # new slope

        # if line search succeeded
        if abs(d3) < -SIG * d0 and f3 < f0 + x3 * RHO * d0:
            X = X + x3 * s  # update variables
            f0 = f3
            fX.append(f0)
            s = (dot(df3, df3) - dot(df0, df3)) / dot(df0, df0) * s - df3
            # Polack-Ribiere CG direction

            df0 = df3  # swap derivatives
            d3 = d0
            d0 = dot(df0, s)
            if d0 > 0:  # new slope must be negative
                s = -df0  # otherwise use steepest direction
                d0 = -dot(s, s)
            # slope ratio but max RATIO
            x3 = x3 * min(RATIO, d3 / (d0 - SMALL))
            ls_failed = 0  # this line search did not fail
        else:
            X = X0  # restore best point so far
            f0 = F0
            df0 = dF0
            # line search failed twice in a row
            if ls_failed or i > abs(length):
                break  # or we ran out of time, so we give up
            s = -df0  # try steepest
            d0 = -dot(s, s)
            x3 = 1 / (1 - d0)
            ls_failed = 1  # this line search failed

    if verbose:
        print '\n'

    return (X, fX, i)


if __name__ == '__main__':
    import numpy as np

    def func(x, theta1, theta2):
        f = ((x - theta1 - theta2) ** 2).sum()
        df = 2 * (x - theta1 - theta2)
        return (f, df)

    X = np.random.normal(0, 1.0, (3, ))
    x0 = X[:]
    theta1 = np.asarray([2.3, 1.1, -4.4])
    theta2 = np.asarray([1., 2., 3.])

    (x, fx, i) = rt_minimize(X, func, -100, theta1, theta2)
    print x0, x, theta1 + theta2, fx[-1], i
