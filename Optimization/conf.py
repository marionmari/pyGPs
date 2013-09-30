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

class random_init_conf(object):
    def __init__(self, mean, cov, lik):
        self.max_trails = None
        self.min_threshold = None
        self.mean = mean
        self.cov = cov
        self.lik = lik
        self._meanRange = [(-10,10) for i in mean.hyp]
        self._covRange  = [(-10,10) for i in cov.hyp]        
        self._likRange  = [(-10,10) for i in lik.hyp]

    def getmr(self):
        return self._meanRange
    def setmr(self, value):
        if len(value) == len(self.mean.hyp):
            self._meanRange = value
        else:
            raise Exception('The length of meanRange is not consistent with number of mean hyparameters')
    meanRange = property(getmr,setmr)
    
    def getcr(self):
        return self._covRange
    def setcr(self, value):
        if len(value) == len(self.cov.hyp):
            self._covRange = value
        else:
            raise Exception('The length of covRange is not consistent with number of covariance hyparameters')
    covRange = property(getcr,setcr)

    def getlr(self):
        return self._likRange
    def setlr(self, value):
        if len(value) == len(self.lik.hyp):
            self._likRange = value
        else:
            raise Exception('The length of likRange is not consistent with number of liklihood hyparameters')
    likRange = property(getlr,setlr)










