from __future__ import division
from __future__ import print_function
from past.utils import old_div
#===============================================================================
#   SCG  Scaled conjugate gradient optimization.
#
#   Copyright (c) Ian T Nabney (1996-2001)
#   updates by Shan Huang 2013
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
#===============================================================================


from math import sqrt
import numpy as np

def run(f, x, args=(), niters = 100, gradcheck = False, display = 0, flog = False, pointlog = False, scalelog = False, tolX = 1.0e-8, tolO = 1.0e-8, eval = None): 
    '''Scaled conjugate gradient optimization. '''
    if display: print('\n***** starting optimization (SCG) *****\n')
    nparams = len(x);
    #  Check gradients
    if gradcheck:
        pass
    eps = 1.0e-4
    sigma0 = 1.0e-4
    result = f(x, *args)
    fold = result[0]             # Initial function value.
    fnow = fold
    funcCount = 1                # Increment function evaluation counter.
    gradnew = result[1]          # Initial gradient.
    gradold = gradnew
    gradCount = 1                # Increment gradient evaluation counter.
    d = -gradnew                 # Initial search direction.
    success = 1                  # Force calculation of directional derivs.
    nsuccess = 0                 # nsuccess counts number of successes.
    beta = 1.0                   # Initial scale parameter.
    betamin = 1.0e-15            # Lower bound on scale.
    betamax = 1.0e50             # Upper bound on scale.
    j = 1                        # j counts number of iterations.
    if flog:
        pass
        #flog(j, :) = fold;
    if pointlog:
        pass
        #pointlog(j, :) = x;

    # Main optimization loop.
    listF = [fold]
    if eval is not None:
        evalue, timevalue = eval(x, *args)
        evalList = [evalue]
        time = [timevalue]

    while (j <= niters):   
        # Calculate first and second directional derivatives.
        if (success == 1):
            mu = np.dot(d, gradnew)
            if (mu >= 0):
                d = - gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            if (kappa < eps):
                #print "FNEW: " , fnow
                #options(8) = fnow
                if eval is not None:
                    return x, listF, evalList, time
                else:
                    return x, listF

            sigma = old_div(sigma0,sqrt(kappa))
            xplus = x + sigma*d
            gplus = f(xplus, *args)[1]
            gradCount += 1
            theta = old_div((np.dot(d, (gplus - gradnew))),sigma);
     
        # Increase effective curvature and evaluate step size alpha.
        delta = theta + beta*kappa
        if (delta <= 0):
            delta = beta*kappa
            beta = beta - old_div(theta,kappa)

        alpha = old_div(- mu,delta)
         
        # Calculate the comparison ratio.
        xnew = x + alpha*d
        fnew = f(xnew, *args)[0]
        funcCount += 1;
        Delta = 2*(fnew - fold)/(alpha*mu)
        if (Delta  >= 0):
            success = 1;
            nsuccess += 1;
            x = xnew;
            fnow = fnew;
            listF.append(fnow)
            if eval is not None:
                evalue, timevalue = eval(x, *args)
                evalList.append(evalue)
                time.append(timevalue)
                
        else:
            success = 0;
            fnow = fold;

        if flog:
            # Store relevant variables
            #flog(j) = fnow;          # Current function value
            pass
        if pointlog:
            #pointlog(j,:) = x;     # Current position
            pass
        if scalelog:
            #scalelog(j) = beta;     # Current scale parameter
            pass
        if display > 0:
            print('***** Cycle %4d  Error %11.6f  Scale %e' %( j, fnow, beta))

        if (success == 1):
        # Test for termination
        # print type (alpha), type(d), type(tolX), type(fnew), type(fold)
            if ((max(abs(alpha*d)) < tolX) & (abs(fnew-fold) < tolO)):
                # options(8) = fnew;
                # print "FNEW: " , fnew
                if eval is not None:
                    return x, listF, evalList, time
                else:
                    return x, listF
            else:
                # Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = f(x, *args)[1]
                gradCount += 1
                # If the gradient is zero then we are done.
                if (np.dot(gradnew, gradnew) == 0):
                    # print "FNEW: " , fnew
                    # options(8) = fnew;
                    if eval is not None:
                        return x, listF, evalList, time
                    else:
                        return x, listF
     
        # Adjust beta according to comparison ratio.
        if (Delta < 0.25):
            beta = min(4.0*beta, betamax);
        if (Delta > 0.75):
            beta = max(0.5*beta, betamin);
     
        # Update search direction using Polak-Ribiere formula, or re-start
        # in direction of negative gradient after nparams steps.
        if (nsuccess == nparams):
            d = -gradnew;
            nsuccess = 0;
        else:
            if (success == 1):
                gamma = old_div(np.dot((gradold - gradnew), gradnew),(mu))
                d = gamma*d - gradnew;

        j += 1
     
    # If we get here, then we haven't terminated in the given number of
    # iterations.
    # options(8) = fold;
    if (display):
        print("maximum number of iterations reached")
    if eval is not None:
        return x, listF, evalList, time
    else:
        return x, listF

