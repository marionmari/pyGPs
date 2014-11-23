Likelihoods & Inference
============================

Changing Likelihood & Inference
------------------------------------
Suggestions of which likelihood and inference method to use is implicitly given
by default,
    * GPR uses Gaussian likelihood and exact inference. 
    * GPC uses Error functionlikelihood and EP inference. 
    * FITC model uses same default with corresponding FITC inference.
    * GPMC calls GPC and thus uses the default setting of GPC

You can change to other likelihood or inference methods using: ::

	model.useLikelihood(newLik)
	model.useInference(newInf)

*newLik* and *newInf* are **Strings**. Currently the options are:
    1. Regression model

       * newLik: **"Laplace"**. Note this will force inference method to be EP.
       * newInf: **"EP"**, **"Laplace"**.

    2. Classification model (including GPMC)

       * newInf: **"Laplace"**

To be consistent with Gaussian Processes community, we use the name "Laplace" for both Laplace likelihood and Laplace inference.
Please note the differences.




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



List of Likelihoods 
---------------------------------------

.. autoclass:: pyGPs.Core.lik.Erf
   :members:

.. autoclass:: pyGPs.Core.lik.Gauss
   :members:

.. autoclass:: pyGPs.Core.lik.Laplace
   :members:

List of Inference 
-----------------------------------------

.. autoclass:: pyGPs.Core.inf.Exact
   :members:

.. autoclass:: pyGPs.Core.inf.EP
   :members:

.. autoclass:: pyGPs.Core.inf.Laplace
   :members:

.. autoclass:: pyGPs.Core.inf.FITC_Exact
   :members:

.. autoclass:: pyGPs.Core.inf.FITC_EP
   :members:

.. autoclass:: pyGPs.Core.inf.FITC_Laplace
   :members:

