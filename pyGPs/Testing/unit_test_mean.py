from __future__ import print_function
from builtins import range
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

class MeanTests(unittest.TestCase):

    def setUp(self):
        # random data for testing
        n = 20     # number of inputs
        D = 3      # dimension of inputs
        self.x = np.random.normal(loc=0.0, scale=1.0, size=(n,D))


    def checkMeanOutput(self, mean):
        n,D = self.x.shape
        self.assertTrue(mean.shape == (n,1))


    def checkDerOutput(self, derivative):
        n,D = self.x.shape
        self.assertTrue(derivative.shape == (n,1))


    def checkMean(self, m):
        mean = m.getMean(x=self.x)             # get mean
        self.checkMeanOutput(mean)
        for der in range(len(m.hyp)):         # get derivatives
            derivative = m.getDerMatrix(x=self.x, der=der)
            self.checkDerOutput(derivative)


    def test_meanZero(self):
        print("testing meanZero...")
        m = pyGPs.mean.Zero()
        self.checkMean(m)



    def test_meanLinear(self):
        print("testing meanLinear...")
        m = pyGPs.mean.Linear(D=self.x.shape[1]) 
        self.checkMean(m)


    def test_meanOne(self):
        print("testing meanOne...")
        m = pyGPs.mean.One() 
        self.checkMean(m)


    def test_meanConst(self):
        print("testing meanConst...")
        m = pyGPs.mean.Const() 
        self.checkMean(m)


    def test_meanScale(self):
        print("testing (compositing mean) muliply by a scalar...")
        m = pyGPs.mean.One() * 6
        self.checkMean(m)   


    def test_meanSum(self):
        print("testing (compositing mean) sum of two means...")
        m = pyGPs.mean.One() + pyGPs.mean.Const() 
        self.checkMean(m)


    def test_meanProduct(self):
        print("testing (compositing mean) product of two means...")
        m = pyGPs.mean.One() * pyGPs.mean.Const() 
        self.checkMean(m)


    def test_meanPower(self):
        print("testing (compositing mean) power of a mean...")
        m = pyGPs.mean.Const() ** 2
        self.checkMean(m)

    # Test your customized mean function
    '''
    def test_mean_new(self):
        # specify your mean function
        self.checkMean(m)
    '''





if __name__ == "__main__":
    print("Running unit tests...")
    unittest.main()

