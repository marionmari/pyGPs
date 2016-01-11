from __future__ import print_function
from builtins import zip
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

class ModelTests(unittest.TestCase):
# note!!
# due to completely random data(which may not make sense for gp in some cases),
# it will rasis error sometimes(failed too much times in multiple-start optimizations)
# therefore we construct toy data reasonably as memtioned in GPML book


    def setUp(self):

        regData = np.load('../Demo/Regression/regression_data.npz')
        self.xr = regData['x']            # training data
        self.yr = regData['y']            # training target
        self.zr = regData['xstar']        # test data
        self.ur = np.array([[-1], [-0.8], [-0.5], [0.3],[1.]]) # inducing points

        clsData = np.load('../Demo/Classification/classification_data.npz')
        self.xc = clsData['x']            # training data
        self.yc = clsData['y']            # training target
        self.zc = clsData['xstar']        # test data
        u1,u2 = np.meshgrid(np.linspace(-2,2,5),np.linspace(-2,2,5))
        self.uc = np.array(list(zip(np.reshape(u2,(np.prod(u2.shape),)),np.reshape(u1,(np.prod(u1.shape),))))) # inducing points


    def checkRegressionOutput(self, model):
        nn,D = self.zr.shape
        self.assertTrue(model.ym.shape == (nn,1))
        self.assertTrue(model.ys2.shape == (nn,1))
        self.assertTrue(model.fm.shape == (nn,1))
        self.assertTrue(model.fs2.shape == (nn,1))


    def checkClassificationOutput(self, model):
        nn,D = self.zc.shape
        self.assertTrue(model.ym.shape == (nn,1))
        self.assertTrue(model.ys2.shape == (nn,1))
        self.assertTrue(model.fm.shape == (nn,1))
        self.assertTrue(model.fs2.shape == (nn,1))


    def test_GPR(self):
        print("testing GP regression...")
        model = pyGPs.GPR()
        m = pyGPs.mean.Zero()
        k = pyGPs.cov.RBF()
        model.setPrior(mean=m, kernel=k)
        model.setOptimizer("Minimize", num_restarts=10)
        model.optimize(self.xr, self.yr)
        model.predict(self.zr)
        self.checkRegressionOutput(model)


    def test_GPC(self):
        print("testing GP classification...")
        model = pyGPs.GPC()
        m = pyGPs.mean.Zero()
        k = pyGPs.cov.RBF()
        model.setPrior(mean=m, kernel=k)
        model.optimize(self.xc, self.yc)
        model.predict(self.zc)
        self.checkClassificationOutput(model)

        model = pyGPs.GPC()
        m = pyGPs.mean.Zero()
        k = pyGPs.cov.RBF()
        model.setPrior(mean=m, kernel=k)
        model.useInference("Laplace")
        model.optimize(self.xc, self.yc)
        model.predict(self.zc)
        self.checkClassificationOutput(model)
        


    def test_GPR_FITC(self):
        print("testing GP sparse regression...")
        model = pyGPs.GPR_FITC()
        m = pyGPs.mean.Zero()
        k = pyGPs.cov.RBF()
        model.setPrior(mean=m, kernel=k, inducing_points=self.ur)
        model.setOptimizer("Minimize", num_restarts=10)
        model.optimize(self.xr, self.yr)
        model.predict(self.zr)
        self.checkRegressionOutput(model)


    def test_GPC_FITC(self):
        print("testing GP sparse classification...")
        model = pyGPs.GPC_FITC()
        m = pyGPs.mean.Zero()
        k = pyGPs.cov.RBF()
        model.setPrior(mean=m, kernel=k, inducing_points=self.uc)
        model.setOptimizer("Minimize", num_restarts=10)
        model.optimize(self.xc, self.yc)
        model.predict(self.zc)
        self.checkClassificationOutput(model)

        model = pyGPs.GPC_FITC()
        m = pyGPs.mean.Zero()
        k = pyGPs.cov.RBF()
        model.setPrior(mean=m, kernel=k, inducing_points=self.uc)
        model.setOptimizer("Minimize", num_restarts=10)
        model.useInference("Laplace")
        model.optimize(self.xc, self.yc)
        model.predict(self.zc)
        self.checkClassificationOutput(model)



if __name__ == "__main__":
    print("Running unit tests(about 5 min)...")
    unittest.main()

