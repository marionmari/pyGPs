Kernels & Means
============================

Simple Kernel & Mean
---------------------
You may already seen, we can specify a kernel function like this(same for mean fucntions): ::
    
    k = pyGPs.cov.RBF( log_ell=-1., log_sigma=0. )

There are several points need to be noticed:

1. Most parameters are initilized in their logorithms. This is because we need to make sure they are positive during optimization. e.g. Here length scale and signal variance should always be positive.

2. Most kernel functions have a scalar in front, namely signal variance(set by log_sigma)

3. If you will do optimization later anyway, you can just leave parameters to be default


Some Special Cases
---------------------
1. For some kernels/means, number of hyperparameters depends on the dimension of input data.
   You can either enter the dimension, which use default values: ::

	   m = pyGPs.mean.Linear( D=x.shape[1] )

   or you can initialze with the exact hyperparameters,
   you should enter as a list, one element for each dimension ::

	   m = pyGPs.mean.Linear( alpha_list=[0.2, 0.4, 0.3] )

   All these "hyp-dim-dependent" functions are:
     * *pyGPs.mean.Linear*
     * *pyGPs.cov.RBFard*
     * *pyGPs.cov.LINard*
     * *pyGPs.cov.RQard*


2. For *pyGPs.cov.RBFunit()*, its signal variance is always 1 (because of unit magnitude). Therefore this function do not have a hyperparameter of "signal variance".


3. *pyGPs.cov.Poly()* has three parameters, where hyperparameters are:
       * c     -> inhomogeneous offset
       * sigma -> signal deviation 
        
   however, 
       * d     -> order of polynomial 
         will be treated as normal parameter, i.e. will not be trained


4. Explicitly set *pyGPs.cov.Noise* is not necessary, because noise are already added in likelihood.


Composite Kernels & Meams 
----------------------------
Adding and muliplying Kernels(Means) is really simple: ::

	k = pyGPs.cov.Periodic() * pyGPs.cov.RBF()
	k = 0.5*pyGPs.cov.LIN() + pyGPs.cov.Periodic()

Scalar will also be treated as a hyperparameter. For example, k = s1 * k1 + s2 * k2, 
then the list of hyperparameters is hyp = [s1, k1.hyp, s2, k2.hyp]. Scalar is passed in logorithm domain such that it will always be positive during optimization.

Beside + / * , there is also a power operator for mean functions: ::

    m = ( pyGPs.mean.One() + pyGPs.mean.Linear(alpha_list=[0.2]) )**2


Precomputed Kernel Matrix
-----------------------------
In certain cases, you may have a precomputed kernel matrix,
but its non-trivial to write down the exact formula of kernel functions. Then you can specify your kernel in the following way. A precomputed kernel also fits with other kernels. In other words, it can also be composited as the way other kernels functions do. ::

	k = pyGPs.cov.Pre(M1, M2)

M1 and M2 are your precomputed kernel matrix,

where,

M1 is a matrix with shape **number of training points plus 1** by **number of test points** 
 - cross covariances matrix (train by test) 
 - last row is self covariances (diagonal of test by test)
M2 is a square matrix with **number of training points** for each dimension
 - training set covariance matrix (train by train)  

A precomputed kernel can also be composited with other kernels. You need to explictly add scalar for *pyGPs.cov.Pre()*. ::
    
    k = 0.5*pyGPs.cov.Pre(M1, M2) + pyGPs.cov.RBF()


Developing New Kernel & Mean Functions
-----------------------------------------
We also support the development of new kernel/mean classes, your customized kernel class need to follow the template as below: ::

    # Your kernel class needs to inherit base class Kernel, 
    # which is in the module of Core.cov
    class MyKernel(Kernel):

      def __init__(self, hyp):
          '''
          Intialize hyperparameters for MyKernel.
          '''
          self.hyp = hyp

      def getCovMatrix(self,x=None,z=None,mode=None):
          '''
          Return the specific covariance matrix according to input mode

          :param x: training data
          :param z: test data
          :param str mode: 'self_test' return self covariance matrix of test data(test by 1). 
                           'train' return training covariance matrix(train by train).
                           'cross' return cross covariance matrix between x and z(train by test)

          :return: the corresponding covariance matrix
          '''
          pass

      def getDerMatrix(self,x=None,z=None,mode=None,der=None):
          '''
          Compute derivatives wrt. hyperparameters according to input mode

          :param x: training data
          :param z: test data
          :param str mode: 'self_test' return self derivative matrix of test data(test by 1). 
                           'train' return training derivative matrix(train by train).
                           'cross' return cross derivative matrix between x and z(train by test)
          :param int der: index of hyperparameter whose derivative to be computed

          :return: the corresponding derivative matrix
          '''
          pass

and for customized mean class: ::

    # Your mean class needs to inherit base class Mean, 
    # which is in the module of Core.mean
    class MyMean(Mean):

      def __init__(self, hyp):
          '''
          Intialize hyperparameters for MyMean.
          '''
          self.hyp = hyp

      def getMean(self, x=None):
          '''
          Get the mean vector.
          '''
          pass

      def getDerMatrix(self, x=None, der=None):
          '''
          Compute derivatives wrt. hyperparameters.

          :param x: training data
          :param int der: index of hyperparameter whose derivative to be computed

          :return: the corresponding derivative matrix
          '''
          pass

You can test your customized mean/kernel function using our framework of unit test. 
Taking kernel test as an example, you can uncomment method *test_cov_new* in 
*pyGPs.Testing.unit_test_cov.py* to check the outputs of your kernel function. ::

    # Test your customized covariance function
    def test_cov_new(self):
        k = myKernel()     # specify your covariance function
        self.checkCovariance(k)

and testing mean function in *pyGPs.Testing.unit_test_mean.py* ::

    # Test your customized mean function
    def test_mean_new(self):
        m = myMean         # specify your mean function
        self.checkMean(m)


