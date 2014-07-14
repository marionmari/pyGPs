Likelihoods & Inference
============================

Changing Likelihood & Inference
------------------------------------
By default,
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

