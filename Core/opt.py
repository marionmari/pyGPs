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
import gp
from scipy.optimize import fmin_bfgs as bfgs
from scipy.optimize import fmin_cg as cg
from ..Optimization import minimize, scg
from copy import deepcopy

class Optimizer(object):
    def __init__(self):
        pass
    def findMin(self):
        pass

    def nlml(self, hypInArray, inffunc, meanfunc, covfunc, likfunc, x, y):
        self.apply_in_objects(hypInArray, meanfunc, covfunc, likfunc)
        result = gp.analyze(inffunc, meanfunc, covfunc, likfunc, x, y, der=False)
        return result[0]

    def dnlml(self, hypInArray, inffunc, meanfunc, covfunc, likfunc, x, y):
        self.apply_in_objects(hypInArray, meanfunc, covfunc, likfunc)
        result = gp.analyze(inffunc, meanfunc, covfunc, likfunc, x, y, der=True)
        dnlml_List = result[1].mean + result[1].cov + result[1].lik
        return np.array(dnlml_List)

    def nlzAnddnlz(self, hypInArray, inffunc, meanfunc, covfunc, likfunc, x, y):
        self.apply_in_objects(hypInArray, meanfunc, covfunc, likfunc)
        result = gp.analyze(inffunc, meanfunc, covfunc, likfunc, x, y, der=True)
        mean_list = result[1].mean
        cov_list = result[1].cov
        lik_list = result[1].lik 
        dnlml_List = mean_list + cov_list + lik_list
        nlZ = result[0]
        return nlZ, np.array(dnlml_List)

    def convert_to_array(self, mean, cov, lik):
        hyplist = mean.hyp + cov.hyp + lik.hyp
        return np.array(hyplist)

    def apply_in_objects(self, hypInArray, mean, cov, lik):
        Lm = len(mean.hyp)
        Lc = len(cov.hyp)
        hypInList = hypInArray.tolist()
        mean.hyp  = hypInList[:Lm]
        cov.hyp   = hypInList[Lm:(Lm+Lc)]
        lik.hyp   = hypInList[(Lm+Lc):]

        
class CG(Optimizer):
    def __init__(self, searchConfig = None):
        self.searchConfig = searchConfig
        self.trailsCounter = 0
        self.errorCounter = 0
    def findMin(self, inffunc, meanfunc, covfunc, likfunc, x, y):
        hypInArray = self.convert_to_array(meanfunc, covfunc, likfunc)
        try:
            opt = cg(self.nlml, hypInArray, self.dnlml, (inffunc, meanfunc, covfunc, likfunc, x, y), maxiter=100, disp=False, full_output=True)
            optimalHyp = deepcopy(opt[0])
            funcValue  = opt[1]
            warnFlag   = opt[4]
            if warnFlag == 1:
                print "Maximum number of iterations exceeded."
            elif warnFlag ==  2:
                print "Gradient and/or function calls not changing."
        except:
            self.errorCounter += 1
            if not self.searchConfig:         
                raise Exception("Can not use conjugate gradient. Try other hyparameters")
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
                    thisopt = cg(self.nlml, hypInArray, self.dnlml, (inffunc, meanfunc, covfunc, likfunc, x, y), maxiter=100, disp=False, full_output=True)
                    if thisopt[1] < funcValue:
                        funcValue  = thisopt[1]
                        optimalHyp = thisopt[0]
                except:
                    self.errorCounter += 1
                if self.searchConfig.num_restarts and self.errorCounter > self.searchConfig.num_restarts/2:
                    print "[CG] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter)
                    raise Exception("Over half of the trails failed for conjugate gradient")
                if self.searchConfig.num_restarts and self.trailsCounter > self.searchConfig.num_restarts-1:         # if exceed num_restarts
                    print "[CG] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter)
                    return optimalHyp, funcValue
                if self.searchConfig.min_threshold and funcValue <= self.searchConfig.min_threshold:           # reach provided mininal
                    print "[CG] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter)
                    return optimalHyp, funcValue 
        return optimalHyp, funcValue


class BFGS(Optimizer):
    def __init__(self, searchConfig = None):
        self.searchConfig = searchConfig
        self.trailsCounter = 0
        self.errorCounter = 0
    def findMin(self,inffunc, meanfunc, covfunc, likfunc, x, y):
        hypInArray = self.convert_to_array(meanfunc, covfunc, likfunc)
        try:
            opt = bfgs(self.nlml, hypInArray, self.dnlml, (inffunc, meanfunc, covfunc, likfunc, x, y), maxiter=100, disp=False, full_output=True)
            optimalHyp = deepcopy(opt[0])
            funcValue  = opt[1]
            warnFlag   = opt[6]
            if warnFlag == 1:
                print "Maximum number of iterations exceeded."
            elif warnFlag ==  2:
                print "Gradient and/or function calls not changing."
        except:
            self.errorCounter += 1
            if not self.searchConfig:         
                raise Exception("Can not use BFGS. Try other hyparameters")
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
                    thisopt = bfgs(self.nlml, hypInArray, self.dnlml, (inffunc, meanfunc, covfunc, likfunc, x, y), maxiter=100, disp=False, full_output=True)
                    if thisopt[1] < funcValue:
                        funcValue  = thisopt[1]
                        optimalHyp = thisopt[0]
                except:
                    self.errorCounter += 1
                if self.searchConfig.num_restarts and self.errorCounter > self.searchConfig.num_restarts/2:
                    print "[BFGS] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter)
                    raise Exception("Over half of the trails failed for BFGS")
                if self.searchConfig.num_restarts and self.trailsCounter > self.searchConfig.num_restarts-1:         # if exceed num_restarts
                    print "[BFGS] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter)
                    return optimalHyp, funcValue
                if self.searchConfig.min_threshold and funcValue <= self.searchConfig.min_threshold:           # reach provided mininal
                    print "[BFGS] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter)
                    return optimalHyp, funcValue 

        return optimalHyp, funcValue


class Minimize(Optimizer):
    def __init__(self, searchConfig = None):
        self.searchConfig = searchConfig
        self.trailsCounter = 0
        self.errorCounter = 0
    def findMin(self,inffunc, meanfunc, covfunc, likfunc, x, y):
        hypInArray = self.convert_to_array(meanfunc, covfunc, likfunc)
        try: 
            opt = minimize.run(self.nlzAnddnlz, hypInArray, (inffunc, meanfunc, covfunc, likfunc, x, y), length=-100)
            optimalHyp = deepcopy(opt[0])
            funcValue  = opt[1][-1]  
        except:
            self.errorCounter += 1
            if not self.searchConfig:         
                raise Exception("Can not use minimize. Try other hyparameters")
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
                    thisopt = minimize.run(self.nlzAnddnlz, hypInArray, (inffunc, meanfunc, covfunc, likfunc, x, y), length=-100)
                    if thisopt[1][-1] < funcValue:
                        funcValue  = thisopt[1][-1]
                        optimalHyp = thisopt[0]
                except:
                    self.errorCounter += 1
                if self.searchConfig.num_restarts and self.errorCounter > self.searchConfig.num_restarts/2:
                    print "[Minimize] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter)
                    raise Exception("Over half of the trails failed for minimize")
                if self.searchConfig.num_restarts and self.trailsCounter > self.searchConfig.num_restarts-1:         # if exceed num_restarts
                    print "[Minimize] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter)
                    return optimalHyp, funcValue
                if self.searchConfig.min_threshold and funcValue <= self.searchConfig.min_threshold:           # reach provided mininal
                    print "[Minimize] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter)
                    return optimalHyp, funcValue                   
        return optimalHyp, funcValue


class SCG(Optimizer):
    def __init__(self, searchConfig = None):
        self.searchConfig = searchConfig
        self.trailsCounter = 0
        self.errorCounter = 0
    def findMin(self,inffunc, meanfunc, covfunc, likfunc, x, y):
        hypInArray = self.convert_to_array(meanfunc, covfunc, likfunc)
        try:
            opt = scg.run(self.nlzAnddnlz, hypInArray, (inffunc, meanfunc, covfunc, likfunc, x, y))
            optimalHyp = deepcopy(opt[0])
            funcValue  = opt[1][-1]
        except:
            self.errorCounter += 1
            if not self.searchConfig:         
                raise Exception("Can not use Scaled conjugate gradient. Try other hyparameters")
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
                    thisopt = scg.run(self.nlzAnddnlz, hypInArray, (inffunc, meanfunc, covfunc, likfunc, x, y))
                    if thisopt[1][-1] < funcValue:
                        funcValue  = thisopt[1][-1]
                        optimalHyp = thisopt[0]
                except:
                    self.errorCounter += 1
                if self.searchConfig.num_restarts and self.errorCounter > self.searchConfig.num_restarts/2:
                    print "[SCG] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter)
                    raise Exception("Over half of the trails failed for Scaled conjugate gradient")
                if self.searchConfig.num_restarts and self.trailsCounter > self.searchConfig.num_restarts-1:         # if exceed num_restarts
                    print "[SCG] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter)
                    return optimalHyp, funcValue
                if self.searchConfig.min_threshold and funcValue <= self.searchConfig.min_threshold:           # reach provided mininal
                    print "[SCG] %d out of %d trails failed during optimization" % (self.errorCounter, self.trailsCounter)
                    return optimalHyp, funcValue 

        return optimalHyp, funcValue


