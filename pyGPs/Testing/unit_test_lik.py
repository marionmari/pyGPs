from __future__ import print_function
from builtins import range
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
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 18/02/2014
#================================================================================

import unittest
import numpy as np
import pyGPs

class LikTests(unittest.TestCase):

    def setUp(self):
        # random data for testing
        self.y = np.random.normal(loc=0.0, scale=1.0, size=(1,1))    # tiled target
        self.mu = np.random.normal(loc=0.0, scale=1.0, size=(1,1))   # tiled conditional mean
        self.s2 = np.random.normal(loc=0.0, scale=1.0, size=(1,1))   # tiled conditional variance


    def checkPrediction(self, lp, ymu, ys2):
        n,D = self.y.shape
        self.assertTrue(lp.shape == (n,D))
        self.assertTrue(ymu.shape == (n,D))
        self.assertTrue(ys2.shape == (n,D))


    def checkInferenceLaplace(self, lp, dlp, d2lp, d3lp):
        n,D = self.y.shape
        self.assertTrue(lp.shape == (n,D))
        self.assertTrue(dlp.shape == (n,D))
        self.assertTrue(d2lp.shape == (n,D))
        self.assertTrue(d3lp.shape == (n,D))

    def checkDerivativeLaplace(self, lp, dlp, d2lp):
        n,D = self.y.shape
        self.assertTrue(lp.shape == (n,D))
        self.assertTrue(dlp.shape == (n,D))
        self.assertTrue(d2lp.shape == (n,D))


    def checkInferenceEP(self, lp, dlp, d2lp):
        n,D = self.y.shape
        self.assertTrue(lp.shape == (n,D))
        self.assertTrue(dlp.shape == (n,D))
        self.assertTrue(d2lp.shape == (n,D))


    def checkDerivativeEP(self, dlZhyp):
        n,D = self.y.shape
        self.assertTrue(dlZhyp.shape == (n,D))


    def checkLikelihood(self, likelihood):
        print("predictive mode")
        lp, ymu, ys2 = likelihood.evaluate(y=self.y, mu=self.mu, s2=self.s2, nargout=3)
        self.checkPrediction(lp, ymu, ys2)

        print("inference mode(Laplace inference)")
        lp,dlp,d2lp,d3lp = likelihood.evaluate(y=self.y, mu=self.mu, s2=self.s2, inffunc=pyGPs.inf.Laplace(), nargout=4)
        self.checkInferenceLaplace(lp,dlp,d2lp,d3lp)
        for der in range(len(likelihood.hyp)):
            lp_dhyp,dlp_dhyp,d2lp_dhyp = likelihood.evaluate(y=self.y, mu=self.mu, s2=self.s2, inffunc=pyGPs.inf.Laplace(), der=der, nargout=4)
            self.checkDerivativeLaplace(lp_dhyp,dlp_dhyp,d2lp_dhyp)

        print("inference mode(EP inference)")
        lp,dlp,d2lp = likelihood.evaluate(y=self.y, mu=self.mu, s2=self.s2, inffunc=pyGPs.inf.EP(), nargout=3)
        self.checkInferenceEP(lp,dlp,d2lp)
        for der in range(len(likelihood.hyp)):
            dlZhyp = likelihood.evaluate(y=self.y, mu=self.mu, s2=self.s2, inffunc=pyGPs.inf.EP(), der=der, nargout=3)
            self.checkDerivativeEP(dlZhyp)


    def test_likGauss(self):
        print("testing Gaussian likelihood...")
        likelihood = pyGPs.lik.Gauss()
        self.checkLikelihood(likelihood)



    def test_likErf(self):
        print("testing error function(cumulative Gaussian) likelihood...")
        likelihood = pyGPs.lik.Erf() 
        self.checkLikelihood(likelihood)


    def test_likLaplace(self):
        print("testing Laplacian likelihood...")
        likelihood = pyGPs.lik.Laplace() 
        self.checkLikelihood(likelihood)


    # Test your customized likelihood function
    '''
    def test_cov_new(self):
        # specify your likelihood function
        self.checkLikelihood(likelihood)
    '''


if __name__ == "__main__":
    print("Running unit tests...")
    unittest.main()

