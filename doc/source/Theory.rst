GPs & Functionality
========================
Gaussian Processes (GPs) can conveniently be used for Bayesian supervised learning, such as regression and classification. 
In its simplest form, GP inference can be implemented in a few lines of code. However, in practice, things typically 
get a little more complicated: you might want to use expressive covariance and mean functions, learn good values 
for hyperparameters, use non-Gaussian likelihood functions (rendering exact inference intractable), use approximate inference 
algorithms, or combinations of many or all of the above. 

A comprehensive introduction to Gaussian Processes for Machine Learning is provided in the `GPML`_ book by Rasmussen and Williams, 2006.



List of Functionality
------------------------

This table lists the functionality implemented in pyGPs. 

+-------------+-------------------+------------+-------------------------+---------------+------------------+
|Functionality| Kernel            | Mean       | Likelihood              | Inference     | Optimizer        |
+=============+===================+============+=========================+===============+==================+
| Simple      | Constant          |  Constant  | Gaussian                | Exact         | Minimize         |
|             +-------------------+------------+-------------------------+---------------+------------------+
|             | Linear            |  Linear    |Cumulative Gaussian (Erf)| EP            | CG               |
|             +-------------------+------------+-------------------------+---------------+------------------+
|             | Linear ard        |  One       | Laplace                 | Laplace       | SCG              |
|             +-------------------+------------+-------------------------+---------------+------------------+
|             | Matérn (1,3,5)    |  Zero      | 	                 |               | BFGS             |
|             +-------------------+------------+                         |               +------------------+
|             | Periodic          |            |                         |               |                  |
|             +-------------------+            |                         |               |                  |  
|             | Polynomial        |            |                         |               |                  |
|             +-------------------+            |                         |               |                  |
|             | Piecewise Poly    |            |                         |               |                  |
|             +-------------------+            |                         |               |                  |
|             | RBF iso           |            |                         |               |                  |
|             +-------------------+            |                         |               |                  |
|             | RBF ard           |            |                         |               |                  |
|             +-------------------+            |                         |               |                  |
|             | RBF unit          |            |                         |               |                  |
|             +-------------------+            |                         |               |                  |
|             | RQ  iso           |            |                         |               |                  |
|             +-------------------+            |                         |               |                  |
|             | RQ  ard           |            |                         |               |                  |
|             +-------------------+            |                         |               |                  |
|             | Noise             |            |                         |               |                  |
+-------------+-------------------+------------+-------------------------+---------------+------------------+
| Composite   | Sum (+)           | Sum (+)    |                         |               |                  |
|             +-------------------+------------+                         |               |                  |
|             | Product  (*)      | Product (*)|                         |               |                  |
|             +-------------------+------------+                         |               |                  |
|             | Scale  (*)        | Scale (*)  |                         |               |                  |
|             +-------------------+------------+                         |               |                  |
|             |                   | Power (**) |                         |               |                  |
+-------------+-------------------+------------+-------------------------+---------------+------------------+
| Sparse GP   |                   |            |                         | FITC Exact    |                  |
|             |                   |            |                         +---------------+                  |
|             |                   |            |                         | FITC EP       |                  |
|             |                   |            |                         +---------------+                  |
|             |                   |            |                         | FITC Laplace  |                  |
+-------------+-------------------+------------+-------------------------+---------------+------------------+
| Graphs      | Diffusion         |            |                         |               |                  |
|             +-------------------+            |                         |               |                  |
|             | VN Diffusion      |            |                         |               |                  |
|             +-------------------+            |                         |               |                  |
|             |Pseudo Inv Laplace |            |                         |               |                  |
|             +-------------------+            |                         |               |                  |
|             |Regularized Laplace|            |                         |               |                  |
|             +-------------------+            |                         |               |                  |
|             |p-step Random Walk |            |                         |               |                  |
|             +-------------------+            |                         |               |                  |
|             |Inverse Cosine     |            |                         |               |                  |
|             +-------------------+            |                         |               |                  |
|             |Propagation Kernel |            |                         |               |                  |
+-------------+-------------------+------------+-------------------------+---------------+------------------+
| Other       | Customized        | Customized |                         |               |                  |
|             +-------------------+------------+                         |               |                  |
|             |Precomputed Kernel |            |                         |               |                  |
+-------------+-------------------+------------+-------------------------+---------------+------------------+

pyGPs also provide cross-validation and some built-in evalation methods.

.. _GPML: http://www.gaussianprocess.org/gpml
