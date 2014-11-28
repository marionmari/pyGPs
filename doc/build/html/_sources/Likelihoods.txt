Likelihoods & Inference
============================

Changing Likelihood & Inference
------------------------------------
Suggestions of which likelihood and inference method to use is implicitly given
by default,
  * GPR uses Gaussian likelihood and exact inference. 
  * GPC uses Erf likelihood and EP inference. 
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








