from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
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

# This is a object-oriented python implementation of gpml functionality
# (Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18).
# based on the functional-version of python implementation
# (Copyright (c) by Marion Neumann and Daniel Marthaler, 20/05/2013)
#
# Copyright (c) by Marion Neumann and Shan Huang, 30/09/2013

import numpy as np
import pyGPs

from scipy.optimize import fmin_bfgs as bfgs
from scipy.optimize import fmin_cg as cg
from pyGPs.Optimization import minimize, scg
from copy import deepcopy

class Optimizer(object):
    def __init__(self, model=None, searchConfig = None):
        self.model = model
        from . import gp

    def findMin(self, x, y, numIters):
        '''
        Find minimal value based on negative-log-marginal-likelihood.
        optimalHyp, funcValue = findMin(x, y, numIters)

        where funcValue is the minimal negative-log-marginal-likelihood during optimization,
        and optimalHyp is a flattened numpy array
        (in sequence of meanfunc.hyp, covfunc.hyp, likfunc.hyp)
        of the hyparameters to achieve such value.

        You can achieve advanced search strategy by initializing Optimizer with searchConfig,
        which is an instance of pyGPs.Optimization.conf.
        See more in pyGPs.Optimization.conf and pyGPs.Core.gp.GP.setOptimizer,
        as well as in online documentation of section Optimizers.
        '''
        pass

    def _nlml(self, hypInArray):
        '''Find negative-log-marginal-likelihood'''
        self._apply_in_objects(hypInArray)
        nlZ, dnlZ = self.model.getPosterior(der=False)
        return nlZ

    def _dnlml(self, hypInArray):
        '''Find derivatives wrt. negative-log-marginal-likelihood'''
        self._apply_in_objects(hypInArray)
        nlZ, dnlZ, post = self.model.getPosterior()
        dnlml_List = dnlZ.mean + dnlZ.cov + dnlZ.lik
        return np.array(dnlml_List)

    def _nlzAnddnlz(self, hypInArray):
        '''Find negative-log-marginal-likelihood and derivatives in one pass(faster)'''
        self._apply_in_objects(hypInArray)
        nlZ, dnlZ, post = self.model.getPosterior()
        dnlml_List = dnlZ.mean + dnlZ.cov + dnlZ.lik
        return nlZ, np.array(dnlml_List)

    def _convert_to_array(self):
        '''Convert all hyparameters in the model to an array'''
        hyplist = self.model.meanfunc.hyp + self.model.covfunc.hyp + self.model.likfunc.hyp
        return np.array(hyplist)

    def _apply_in_objects(self, hypInArray):
        '''Apply the values in the input array to hyparameters of model.'''
        Lm = len(self.model.meanfunc.hyp)
        Lc = len(self.model.covfunc.hyp)
        hypInList = hypInArray.tolist()
        self.model.meanfunc.hyp  = hypInList[:Lm]
        self.model.covfunc.hyp   = hypInList[Lm:(Lm+Lc)]
        self.model.likfunc.hyp   = hypInList[(Lm+Lc):]


class CG(Optimizer):
    '''Conjugent gradient'''
    def __init__(self, model, searchConfig = None):
        super(CG, self).__init__()
        self.model = model
        self.searchConfig = searchConfig
        self.trailsCounter = 0
        self.errorCounter = 0

    def findMin(self, x, y, numIters = 100):
        meanfunc = self.model.meanfunc
        covfunc = self.model.covfunc
        likfunc = self.model.likfunc
        inffunc = self.model.inffunc
        hypInArray = self._convert_to_array()
        try:
            opt = cg(self._nlml, hypInArray, self._dnlml, maxiter=numIters, disp=False, full_output=True)
            optimalHyp = deepcopy(opt[0])
            funcValue  = opt[1]
            warnFlag   = opt[4]
            if warnFlag == 1:
                print("Maximum number of iterations exceeded.")
            elif warnFlag ==  2:
                print("Gradient and/or function calls not changing.")
        except:
            self.errorCounter += 1
            if not self.searchConfig:         
                raise Exception("Can not learn hyperparamters using conjugate gradient.")
        self.trailsCounter += 1

        if self.searchConfig:
            searchRange = self.searchConfig.meanRange + self.searchConfig.covRange + self.searchConfig.likRange 
            if not (self.searchConfig.num_restarts or self.searchConfig.min_threshold):
                raise Exception('Specify at least one of the stop conditions')
            while True:
                self.trailsCounter += 1                 # increase counter
                for i in range(hypInArray.shape[0]):   # random init of hyp
                    hypInArray[i]= np.random.uniform(low=searchRange[i][0], high=searchRange[i][1])
                # value this time is better than optiaml min value
                try:
                    thisopt = cg(self._nlml, hypInArray, self._dnlml, maxiter=100, disp=False, full_output=True)
                    if thisopt[1] < funcValue:
                        funcValue  = thisopt[1]
                        optimalHyp = thisopt[0]
                except:
                    self.errorCounter += 1
                if self.searchConfig.num_restarts and self.errorCounter > old_div(self.searchConfig.num_restarts,2):
                    print("[CG] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    raise Exception("Over half of the trails failed for conjugate gradient")
                if self.searchConfig.num_restarts and self.trailsCounter > self.searchConfig.num_restarts-1:         # if exceed num_restarts
                    print("[CG] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    return optimalHyp, funcValue
                if self.searchConfig.min_threshold and funcValue <= self.searchConfig.min_threshold:           # reach provided mininal
                    print("[CG] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    return optimalHyp, funcValue 
        return optimalHyp, funcValue



class BFGS(Optimizer):
    '''quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)'''
    def __init__(self, model, searchConfig = None):
        super(BFGS, self).__init__()
        self.model = model
        self.searchConfig = searchConfig
        self.trailsCounter = 0
        self.errorCounter = 0

    def findMin(self, x, y, numIters = 100):
        meanfunc = self.model.meanfunc
        covfunc = self.model.covfunc
        likfunc = self.model.likfunc
        inffunc = self.model.inffunc
        hypInArray = self._convert_to_array()

        try:
            opt = bfgs(self._nlml, hypInArray, self._dnlml, maxiter=numIters, disp=False, full_output=True)
            optimalHyp = deepcopy(opt[0])
            funcValue  = opt[1]
            warnFlag   = opt[6]
            if warnFlag == 1:
                print("Maximum number of iterations exceeded.")
            elif warnFlag ==  2:
                print("Gradient and/or function calls not changing.")
        except:
            self.errorCounter += 1
            if not self.searchConfig:         
                raise Exception("Can not learn hyperparamters using BFGS.")
        self.trailsCounter += 1


        if self.searchConfig:
            searchRange = self.searchConfig.meanRange + self.searchConfig.covRange + self.searchConfig.likRange 
            if not (self.searchConfig.num_restarts or self.searchConfig.min_threshold):
                raise Exception('Specify at least one of the stop conditions')
            while True:
                self.trailsCounter += 1                 # increase counter
                for i in range(hypInArray.shape[0]):   # random init of hyp
                    hypInArray[i]= np.random.uniform(low=searchRange[i][0], high=searchRange[i][1])
                # value this time is better than optiaml min value
                try:
                    thisopt = bfgs(self._nlml, hypInArray, self._dnlml, maxiter=100, disp=False, full_output=True)
                    if thisopt[1] < funcValue:
                        funcValue  = thisopt[1]
                        optimalHyp = thisopt[0]
                except:
                    self.errorCounter += 1
                if self.searchConfig.num_restarts and self.errorCounter > old_div(self.searchConfig.num_restarts,2):
                    print("[BFGS] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    raise Exception("Over half of the trails failed for BFGS")
                if self.searchConfig.num_restarts and self.trailsCounter > self.searchConfig.num_restarts-1:         # if exceed num_restarts
                    print("[BFGS] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    return optimalHyp, funcValue
                if self.searchConfig.min_threshold and funcValue <= self.searchConfig.min_threshold:           # reach provided mininal
                    print("[BFGS] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    return optimalHyp, funcValue

        return optimalHyp, funcValue



class Minimize(Optimizer):
    '''minimize by Carl Rasmussen (python implementation of "minimize" in GPML)'''
    def __init__(self, model, searchConfig = None):
        super(Minimize, self).__init__()
        self.model = model
        self.searchConfig = searchConfig
        self.trailsCounter = 0
        self.errorCounter = 0

    def findMin(self, x, y, numIters = 200):
        meanfunc = self.model.meanfunc
        covfunc = self.model.covfunc
        likfunc = self.model.likfunc
        inffunc = self.model.inffunc
        hypInArray = self._convert_to_array()

        try:
            # opt = minimize.run(self._nlzAnddnlz, hypInArray, length=-numIters)
            opt = minimize.run(self._nlzAnddnlz, hypInArray, length=numIters)
            optimalHyp = deepcopy(opt[0])
            funcValue  = opt[1][-1]
            print("Number of line searches %g" % opt[2])
        except:
            self.errorCounter += 1
            if not self.searchConfig:
                raise Exception("Can not learn hyperparamters using minimize.")
        self.trailsCounter += 1

        if self.searchConfig:
            searchRange = self.searchConfig.meanRange + self.searchConfig.covRange + self.searchConfig.likRange
            if not (self.searchConfig.num_restarts or self.searchConfig.min_threshold):
                raise Exception('Specify at least one of the stop conditions')
            while True:
                self.trailsCounter += 1                 # increase counter
                for i in xrange(hypInArray.shape[0]):   # random init of hyp
                    hypInArray[i]= np.random.uniform(low=searchRange[i][0], high=searchRange[i][1])
                # value this time is better than optiaml min value
                try:
                    # thisopt = minimize.run(self._nlzAnddnlz, hypInArray, length=-numIters)
                    thisopt = minimize.run(self._nlzAnddnlz, hypInArray, length=numIters)
                    print("Number of line searches %g" % thisopt[2])
                    if thisopt[1][-1] < funcValue:
                        funcValue  = thisopt[1][-1]
                        optimalHyp = thisopt[0]
                except:
                    self.errorCounter += 1
                if self.searchConfig.num_restarts and self.errorCounter > self.searchConfig.num_restarts/2:
                    print("[Minimize] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    raise Exception("Over half of the trails failed for minimize")
                if self.searchConfig.num_restarts and self.trailsCounter > self.searchConfig.num_restarts-1:         # if exceed num_restarts
                    print("[Minimize] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    return optimalHyp, funcValue
                if self.searchConfig.min_threshold and funcValue <= self.searchConfig.min_threshold:           # reach provided mininal
                    print("[Minimize] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    return optimalHyp, funcValue
        return optimalHyp, funcValue



class SCG(Optimizer):
    '''Scaled conjugent gradient (faster than CG)'''
    def __init__(self, model, searchConfig = None):
        super(SCG, self).__init__()
        self.model = model
        self.searchConfig = searchConfig
        self.trailsCounter = 0
        self.errorCounter = 0

    def findMin(self, x, y, numIters = 100):
        meanfunc = self.model.meanfunc
        covfunc = self.model.covfunc
        likfunc = self.model.likfunc
        inffunc = self.model.inffunc
        hypInArray = self._convert_to_array()
        try:
            opt = scg.run(self._nlzAnddnlz, hypInArray, niters = numIters)
            optimalHyp = deepcopy(opt[0])
            funcValue  = opt[1][-1]
        except:
            self.errorCounter += 1
            if not self.searchConfig:
                raise Exception("Can not learn hyperparamters using Scaled conjugate gradient.")
        self.trailsCounter += 1

        if self.searchConfig:
            searchRange = self.searchConfig.meanRange + self.searchConfig.covRange + self.searchConfig.likRange
            if not (self.searchConfig.num_restarts or self.searchConfig.min_threshold):
                raise Exception('Specify at least one of the stop conditions')
            while True:
                self.trailsCounter += 1                 # increase counter
                for i in range(hypInArray.shape[0]):   # random init of hyp
                    hypInArray[i]= np.random.uniform(low=searchRange[i][0], high=searchRange[i][1])
                # value this time is better than optiaml min value
                try:
                    thisopt = scg.run(self._nlzAnddnlz, hypInArray)
                    if thisopt[1][-1] < funcValue:
                        funcValue  = thisopt[1][-1]
                        optimalHyp = thisopt[0]
                except:
                    self.errorCounter += 1
                if self.searchConfig.num_restarts and self.errorCounter > old_div(self.searchConfig.num_restarts,2):
                    print("[SCG] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    raise Exception("Over half of the trails failed for Scaled conjugate gradient")
                if self.searchConfig.num_restarts and self.trailsCounter > self.searchConfig.num_restarts-1:         # if exceed num_restarts
                    print("[SCG] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    return optimalHyp, funcValue
                if self.searchConfig.min_threshold and funcValue <= self.searchConfig.min_threshold:           # reach provided mininal
                    print("[SCG] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter))
                    return optimalHyp, funcValue 

        return optimalHyp, funcValue



