from __future__ import print_function
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

import unittest
import numpy as np
import pyGPs

class OptTests(unittest.TestCase):

    def setUp(self):
        # fix random seed
        np.random.seed(0)
        
        # random data for testing
        n = 20     # number of inputs
        D = 3      # dimension of inputs
        self.x = np.random.normal(loc=0.0, scale=1.0, size=(n,D))
        self.y = np.random.random((n,))
        self.model = pyGPs.GPR()
        nlZ, dnlZ, post = self.model.getPosterior(self.x,self.y)
        self.nlZ_beforeOpt = nlZ



    def checkOptimizer(self, optimizer):
        # funcValue is the minimal negative-log-marginal-likelihood during optimization,
        # and optimalHyp is a flattened numpy array
        optimalHyp, funcValue = optimizer.findMin(self.x, self.y)
        self.assertTrue(funcValue < self.nlZ_beforeOpt)     
        print("optimal hyperparameters in flattened array:", optimalHyp) 



    def test_CG(self):
        print("testing Conjugent gradient...")
        optimizer = pyGPs.Core.opt.CG(self.model)
        self.checkOptimizer(optimizer)



    def test_BFGS(self):
        print("testing quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)...")
        optimizer = pyGPs.Core.opt.BFGS(self.model)
        self.checkOptimizer(optimizer)



    def test_SCG(self):
        print("testing Scaled conjugent gradient...")
        optimizer = pyGPs.Core.opt.SCG(self.model)
        self.checkOptimizer(optimizer)



    def test_Minimize(self):
        print("testing minimize by Carl Rasmussen ...")
        optimizer = pyGPs.Core.opt.Minimize(self.model)
        self.checkOptimizer(optimizer)



    # Test your customized mean function
    '''
    def test_MyOptimizer(self):
        # specify your mean function
        optimizer = pyGPs.Core.opt.MyOptimizer(self.model)
        self.checkOptimizer(optimizer)
    '''




if __name__ == "__main__":
    print("Running unit tests...")
    unittest.main()

