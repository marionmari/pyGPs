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

class InfTests(unittest.TestCase):
# here focus on testing inference function.
# therefore only use one example set of covariance/mean functions

    def setUp(self):
        # random 2d data for testing
        self.x = np.random.normal(loc=0.0, scale=1.0, size=(20,2))
        self.y = np.random.normal(loc=0.0, scale=1.0, size=(20,1))
        self.u = np.random.random(size=(5,2))        # random inducing points


    def checkInferenceOutput(self, post, nlZ, dnlZ):
        n,D = self.x.shape
        self.assertTrue(post.alpha.shape[0] == n)
        self.assertTrue(post.L.shape == (n,n))
        self.assertTrue(post.sW.shape == (n,1))
        self.assertTrue(type(nlZ) is np.float64)
        for entry in dnlZ.mean:
            self.assertTrue(type(entry) is np.float64)
        for entry in dnlZ.cov:
            self.assertTrue(type(entry) is np.float64)
        for entry in dnlZ.lik:
            self.assertTrue(type(entry) is np.float64)


    def checkFITCOutput(self, post, nlZ, dnlZ):
        n,D = self.x.shape
        nu,D = self.u.shape
        self.assertTrue(post.alpha.shape[0] == nu)
        self.assertTrue(post.L.shape == (nu,nu))
        self.assertTrue(post.sW.shape == (n,1))
        self.assertTrue(type(nlZ) is np.float64)
        for entry in dnlZ.mean:
            self.assertTrue(type(entry) is np.float64)
        for entry in dnlZ.cov:
            self.assertTrue(type(entry) is np.float64)
        for entry in dnlZ.lik:
            self.assertTrue(type(entry) is np.float64)


    def test_infExact(self):
        print("testing exact inference...")
        inffunc = pyGPs.inf.Exact()
        meanfunc = pyGPs.mean.Zero()
        covfunc = pyGPs.cov.RBF()
        likfunc = pyGPs.lik.Gauss()
        post, nlZ, dnlZ = inffunc.evaluate(meanfunc, covfunc, likfunc, self.x, self.y, nargout=3)
        self.checkInferenceOutput(post, nlZ, dnlZ)


    def test_infFITC_Exact(self):
        print("testing FITC inference...")
        inffunc = pyGPs.inf.FITC_Exact()
        meanfunc = pyGPs.mean.Zero()
        covfunc = pyGPs.cov.RBF().fitc(self.u)
        likfunc = pyGPs.lik.Gauss()
        post, nlZ, dnlZ = inffunc.evaluate(meanfunc, covfunc, likfunc, self.x, self.y, nargout=3)
        self.checkFITCOutput(post, nlZ, dnlZ)


    def test_infEP(self):
        print("testing EP inference...")
        inffunc = pyGPs.inf.EP()
        meanfunc = pyGPs.mean.Zero()
        covfunc = pyGPs.cov.RBF()
        likfunc = pyGPs.lik.Gauss()
        post, nlZ, dnlZ = inffunc.evaluate(meanfunc, covfunc, likfunc, self.x, self.y, nargout=3)
        self.checkInferenceOutput(post, nlZ, dnlZ)


    def test_infFITC_EP(self):
        print("testing FITC EP inference...")
        inffunc = pyGPs.inf.FITC_EP()
        meanfunc = pyGPs.mean.Zero()
        covfunc = pyGPs.cov.RBF().fitc(self.u)
        likfunc = pyGPs.lik.Gauss()
        post, nlZ, dnlZ = inffunc.evaluate(meanfunc, covfunc, likfunc, self.x, self.y, nargout=3)
        self.checkFITCOutput(post, nlZ, dnlZ)


    def test_infLaplace(self):
        print("testing Laplace inference...")
        inffunc = pyGPs.inf.Laplace()
        meanfunc = pyGPs.mean.Zero()
        covfunc = pyGPs.cov.RBF()
        likfunc = pyGPs.lik.Gauss()
        post, nlZ, dnlZ = inffunc.evaluate(meanfunc, covfunc, likfunc, self.x, self.y, nargout=3)
        self.checkInferenceOutput(post, nlZ, dnlZ)


    def test_infFITC_Laplace(self):
        print("testing FITC EP inference...")
        inffunc = pyGPs.inf.FITC_Laplace()
        meanfunc = pyGPs.mean.Zero()
        covfunc = pyGPs.cov.RBF().fitc(self.u)
        likfunc = pyGPs.lik.Gauss()
        post, nlZ, dnlZ = inffunc.evaluate(meanfunc, covfunc, likfunc, self.x, self.y, nargout=3)
        self.checkFITCOutput(post, nlZ, dnlZ)
    
    # Test your customized inference function
    '''
    def test_inf_new(self):
        # specify your inf function
        # set mean/cov/lik functions
        post, nlZ, dnlZ = inffunc.evaluate(meanfunc, covfunc, likfunc, self.x, self.y, nargout=3)
        self.checkFITCOutput(post, nlZ, dnlZ)
    '''



if __name__ == "__main__":
    print("Running unit tests...")
    unittest.main()

