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


import numpy as np

class random_init_conf(object):
    def __init__(self, mean, cov, lik):
        self.num_restarts = None
        self.min_threshold = None
        self.mean = mean
        self.cov = cov
        self.lik = lik
        self._meanRange = [(-5,5) for i in mean.hyp]
        self._covRange  = [(-5,5) for i in cov.hyp]        
        self._likRange  = [(-5,5) for i in lik.hyp]

    def _getmr(self):
        return self._meanRange
    def _setmr(self, value):
        if len(value) == len(self.mean.hyp):
            self._meanRange = value
        else:
            raise Exception('The length of meanRange is not consistent with number of mean hyparameters')
    meanRange = property(_getmr,_setmr)
    
    def _getcr(self):
        return self._covRange
    def _setcr(self, value):
        if len(value) == len(self.cov.hyp):
            self._covRange = value
        else:
            raise Exception('The length of covRange is not consistent with number of covariance hyparameters')
    covRange = property(_getcr,_setcr)

    def _getlr(self):
        return self._likRange
    def _setlr(self, value):
        if len(value) == len(self.lik.hyp):
            self._likRange = value
        else:
            raise Exception('The length of likRange is not consistent with number of liklihood hyparameters')
    likRange = property(_getlr,_setlr)










