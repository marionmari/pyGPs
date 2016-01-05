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
from pyGPs.Core.tools import jitchol

class CovarianceTests(unittest.TestCase):

    def setUp(self):
        # fix random seed
        np.random.seed(0)
        # random data for testing
        n = 20          # number of training inputs
        nn = 10         # number of test inputs
        D = 2           # dimension of inputs 
        self.x = np.random.normal(loc=0.0, scale=20.0, size=(n,D))
        self.z = np.random.normal(loc=0.0, scale=20.0, size=(nn,D))
        # random precomputed kernel matrix
        self.M1 = np.random.random(size=(21,10))
        self.M2 = np.random.random(size=(20,20))


    def checkCovOutput(self, train_train, train_test, self_test):
        n,D = self.x.shape
        nn, D = self.z.shape
        self.assertTrue(train_train.shape == (n,n))
        self.assertTrue(train_test.shape == (n,nn))
        self.assertTrue(self_test.shape == (nn,1))

        # check eigenvalues of kernele matrix
        self.assertTrue(self.is_positive_semi_definite(train_train))   

        # check Cholesky decomposition of kernel matrix
        # with jitter added to the diagnal for numerical if necessary
        # if the kernel matrixes can compute Cholesky decomposition withour errors,
        # it is positive definite up to numerical precision (up to the added jitters). 
        jitchol(train_train)


    def is_positive_semi_definite(self, K):
        v = np.linalg.eig(K)[0]
        if any(v.real<=-1e-9):
            print(v.real.min())
            return False
        else:
            return True


    def checkDerOutput(self, train_train, train_test, self_test):
        n,D = self.x.shape
        nn, D = self.z.shape
        self.assertTrue(train_train.shape == (n,n))
        self.assertTrue(train_test.shape == (n,nn))
        self.assertTrue(self_test.shape == (nn,1))


    def checkCovariance(self, k):
        k1 = k.getCovMatrix(x=self.x, mode='train')           # test train by train covariance
        k2 = k.getCovMatrix(x=self.x, z=self.z, mode='cross') # test train by test covariance
        k3 = k.getCovMatrix(z=self.z, mode='self_test')       # test test by test self covariance
        self.checkCovOutput(k1,k2,k3)
        for der in range(len(k.hyp)):                        # checking derivatives for each hyperparameter
            kd1 = k.getDerMatrix(x=self.x, mode='train',der=der)           # test train by train derivative
            kd2 = k.getDerMatrix(x=self.x, z=self.z, mode='cross',der=der) # test train by test derivative
            kd3 = k.getDerMatrix(z=self.z, mode='self_test',der=der)       # test test by test self derivative
            self.checkDerOutput(kd1, kd2, kd3)


    def test_covSM(self):
        print("testing covSM...")
        k = pyGPs.cov.SM(Q=10,D=self.x.shape[1])
        self.checkCovariance(k)

    def test_covGabor(self):
        print("testing covGabor...")
        k = pyGPs.cov.Gabor()
        self.checkCovariance(k)

    def test_covRBF(self):
        print("testing covRBF...")
        k = pyGPs.cov.RBF()
        self.checkCovariance(k)


    def test_covRBFunit(self):
        print("testing covRBFunit...")
        k = pyGPs.cov.RBFunit()
        self.checkCovariance(k)


    def test_covRBFard(self):
        print("testing covRBFard...")
        k = pyGPs.cov.RBFard(D=self.x.shape[1])
        self.checkCovariance(k)


    def test_covConst(self):
        print("testing covConst...")
        k = pyGPs.cov.Const()
        self.checkCovariance(k)


    def test_covLinear(self):
        print("testing covLinear...")
        k = pyGPs.cov.Linear()
        self.checkCovariance(k)


    def test_covLINard(self):
        print("testing covLINard...")
        k = pyGPs.cov.LINard(D=self.x.shape[1])
        self.checkCovariance(k)


    def test_covMatern(self):
        print("testing covMatern...")
        print("...d = 1...")
        k = pyGPs.cov.Matern(d=1)
        self.checkCovariance(k)

        print("testing covMatern...")
        print("...d = 3...")
        k = pyGPs.cov.Matern(d=3)
        self.checkCovariance(k)

        print("testing covMatern...")
        print("...d = 5...")
        k = pyGPs.cov.Matern(d=5)
        self.checkCovariance(k)

        print("testing covMatern...")
        print("...d = 7...")
        k = pyGPs.cov.Matern(d=7)
        self.checkCovariance(k)


    def test_covPeriodic(self):
        print("testing covPeriodic...")
        k = pyGPs.cov.Periodic()
        n = 20          # number of training inputs
        nn = 10         # number of test inputs
        D = 1           # Note Periodic covariance can only use 1d data
        x = np.random.normal(loc=0.0, scale=20.0, size=(n,D))
        z = np.random.normal(loc=0.0, scale=20.0, size=(nn,D))
        k1 = k.getCovMatrix(x=x, mode='train')           # test train by train covariance
        k2 = k.getCovMatrix(x=x, z=z, mode='cross') # test train by test covariance
        k3 = k.getCovMatrix(z=z, mode='self_test')       # test test by test self covariance
        self.assertTrue(k1.shape == (n,n))
        self.assertTrue(k2.shape == (n,nn))
        self.assertTrue(k3.shape == (nn,1))
        for der in range(len(k.hyp)):                        # checking derivatives for each hyperparameter
            kd1 = k.getDerMatrix(x=x, mode='train',der=der)           # test train by train derivative
            kd2 = k.getDerMatrix(x=x, z=z, mode='cross',der=der) # test train by test derivative
            kd3 = k.getDerMatrix(z=z, mode='self_test',der=der)       # test test by test self derivative
            self.checkDerOutput(kd1, kd2, kd3)



    def test_covNoise(self):
        print("testing covNoise...")
        k = pyGPs.cov.Noise()
        self.checkCovariance(k)


    def test_covRQ(self):
        print("testing covRQ...")
        k = pyGPs.cov.RQ()
        self.checkCovariance(k)


    def test_covRQard(self):
        print("testing covRQard...")
        k = pyGPs.cov.RQard(D=self.x.shape[1])
        self.checkCovariance(k)


    def test_covPre(self):
        print("testing covPre...")
        k = pyGPs.cov.Pre(self.M1, self.M2)                   # load precomputed kernel matrix
        k1 = k.getCovMatrix(x=self.x, mode='train')           # test train by train covariance
        k2 = k.getCovMatrix(x=self.x, z=self.z, mode='cross') # test train by test covariance
        k3 = k.getCovMatrix(z=self.z, mode='self_test')       # test test by test self covariance
        n,D = self.x.shape
        nn, D = self.z.shape
        self.assertTrue(k1.shape == (n,n))
        self.assertTrue(k2.shape == (n,nn))
        self.assertTrue(k3.shape == (nn,1))
        for der in range(len(k.hyp)):                        # checking derivatives for each hyperparameter
            kd1 = k.getDerMatrix(x=self.x, mode='train',der=der)           # test train by train derivative
            kd2 = k.getDerMatrix(x=self.x, z=self.z, mode='cross',der=der) # test train by test derivative
            kd3 = k.getDerMatrix(z=self.z, mode='self_test',der=der)       # test test by test self derivative
            self.checkDerOutput(kd1, kd2, kd3)


    def test_covPiecePoly(self):
        print("testing covPiecePoly...")
        k = pyGPs.cov.PiecePoly()
        self.checkCovariance(k)


    def test_covPoly(self):
        print("testing covPoly...")
        k = pyGPs.cov.Poly()
        self.checkCovariance(k)


    def test_covScale(self):
        print("testing (composing kernel) muliply by a scalar...")
        k = pyGPs.cov.RBF()*5
        self.checkCovariance(k)


    def test_covSum(self):
        print("testing (composing kernel) sum of two kernels...")
        k = pyGPs.cov.RBF() + pyGPs.cov.PiecePoly()
        self.checkCovariance(k)


    def test_covProduct(self):
        print("testing (composing kernel) product of two kernels...")
        k = pyGPs.cov.RBF() * pyGPs.cov.PiecePoly()
        self.checkCovariance(k)


    def test_covFITC(self):
        print("testing FITC kernel to be used with sparse GP...")
        n,D  = self.x.shape
        nn,D = self.z.shape
        nu = 5        # number of inducing points
        u = np.random.random(size=(nu,D))                      # random inducing points
        
        k = pyGPs.cov.RBF().fitc(u)

        K, Kuu, Ku = k.getCovMatrix(x=self.x, mode='train')   # test train by train covariance
        k2 = k.getCovMatrix(x=self.x, z=self.z, mode='cross') # test train by test covariance
        k3 = k.getCovMatrix(z=self.z, mode='self_test')       # test test by test self covariance
        self.assertTrue(K.shape == (n,1))
        self.assertTrue(Kuu.shape == (nu,nu))
        self.assertTrue(Ku.shape == (nu,n))
        self.assertTrue(k2.shape == (nu,nn))
        self.assertTrue(k3.shape == (nn,1))

        for der in range(len(k.hyp)):
            Kd, Kuud, Kud = k.getDerMatrix(x=self.x, mode='train',der=der) # test train by train derivative
            kd2 = k.getDerMatrix(x=self.x, z=self.z, mode='cross',der=der) # test train by test derivative
            kd3 = k.getDerMatrix(z=self.z, mode='self_test',der=der)       # test test by test self derivative
            self.assertTrue(Kd.shape == (n,1))
            self.assertTrue(Kuud.shape == (nu,nu))
            self.assertTrue(Kud.shape == (nu,n))
            self.assertTrue(kd2.shape == (nu,nn))
            self.assertTrue(kd3.shape == (nn,1))


    # Test your customized covariance function
    '''
    def test_cov_new(self):
        k = myKernel()     # specify your covariance function
        self.checkCovariance(k)
    '''

if __name__ == "__main__":
    print("Running unit tests...")
    unittest.main()

