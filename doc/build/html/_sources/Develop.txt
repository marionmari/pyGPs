Developing Customized Functions
======================================


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



Developing New Likelihood & Inference Functions
-------------------------------------------------
We also support the development of new likelihood/inference classes, your customized inference class need to follow the template as below: ::

    # Your inference class needs to inherit base class Inference, 
    # which is in the module of Core.inf
    class MyInference(Kernel):

      def __init__(self):
          pass

      def evaluate(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
          '''
          Inference computation based on inputs.
          post, nlZ, dnlZ = inffunc.evaluate(mean, cov, lik, x, y)

          INPUT:
          cov: name of the covariance function (see covFunctions.m)
          lik: name of the likelihood function (see likFunctions.m)
          x: n by D matrix of training inputs 
          y: 1d array (of size n) of targets

          OUTPUT:
          post(instance of postStruct): struct representation of the (approximate) posterior containing: 
          nlZ: returned value of the negative log marginal likelihood
          dnlZ(instance of dnlZStruct): struct representation for derivatives of the negative log marginal likelihood
          w.r.t. each hyperparameter.

          Usually, the approximate posterior to be returned admits the form: 
          N(m=K*alpha, V=inv(inv(K)+W)), where alpha is a vector and W is diagonal;
          if not, then L contains instead -inv(K+inv(W)), and sW is unused.

          For more information on the individual approximation methods and their
          implementations, see the respective inference function below. See also gp.py

          :param meanfunc: mean function
          :param covfunc: covariance function
          :param likfunc: likelihood function
          :param x: training data
          :param y: training labels
          :param nargout: specify the number of output(1,2 or 3)
          :return: posterior, negative-log-marginal-likelihood, derivative for negative-log-marginal-likelihood-likelihood
          '''
          pass

where **postStruct** and **dnlZStruct** is also defined in *Core.inf*. ::

    class postStruct(object):
        '''
        Data structure for posterior

        post.alpha ->  1d array containing inv(K)*m, 
                       where K is the prior covariance matrix and m the approx posterior mean
        post.sW:   ->  1d array containing diagonal of sqrt(W)
                       the approximate posterior covariance matrix is inv(inv(K)+W)
        post.L     ->  2d array, L = chol(sW*K*sW+identity(n))
        '''

    class dnlZStruct(object):
        '''
        Data structure for the derivatives of mean, cov and lik functions.

        dnlZ.mean  ->  list of derivatives for each hyperparameters in mean function
        dnlZ.cov   ->  list of derivatives for each hyperparameters in covariance function
        dnlZ.lik   ->  list of derivatives for each hyperparameters in likelihood function
        '''


Customizing likelihood function is more complicated. We will omit it here to keep this this page not too long.
However, you can find detailed explaination either in the **source code** *Core.lik* or in coresponding section of **manual**.

Just like testing kernel/mean fucntions, you can also find unit test module for likelihood and inference functions.
To test your customized inference function, uncomment the following method in *pyGPs.Testing.unit_test_inf.py*. ::

    # Test your customized inference function
    def test_inf_new(self):
        # specify your inf function
        # set mean/cov/lik functions
        post, nlZ, dnlZ = inffunc.evaluate(meanfunc, covfunc, likfunc, self.x, self.y, nargout=3)
        self.checkFITCOutput(post, nlZ, dnlZ)

and test customized likelihood function in *pyGPs.Testing.unit_test_lik.py* ::

    # Test your customized likelihood function
    def test_cov_new(self):
        likelihood = myLikelihood()     # specify your likelihood function
        self.checkLikelihood(likelihood)



Developing Optimization Methods
-------------------------------------
We also support the development of new optimizers. 

Your own optimizer should inherent base class Optimizer in *pyGPs.Core.opt*
and follow the template as below: ::

    class MyOptimizer(Optimizer):
        def __init__(self, model=None, searchConfig = None):
            self.model = model

        def findMin(self, x, y):
            '''
            Find minimal value based on negative-log-marginal-likelihood. 
            optimalHyp, funcValue = findMin(x, y)

            where funcValue is the minimal negative-log-marginal-likelihood during optimization,
            and optimalHyp is a flattened numpy array 
            (in sequence of meanfunc.hyp, covfunc.hyp, likfunc.hyp) 
            of the hyparameters to achieve such value.

            You can achieve advanced search strategy by initializing Optimizer with searchConfig, 
            which is an instance of pyGPs.Optimization.conf. 
            See more in pyGPs.Optimization.conf and pyGPs.Core.gp.GP.setOptimizer, 
            as well as in online documentation of section Optimizers.
            '''
            pass








