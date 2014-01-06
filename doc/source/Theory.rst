GPs & Functionality
========================
Gaussian Processes (GPs) can conveniently be used for Bayesian supervised learning, such as regression and classification. 
In its simplest form, GP inference can be implemented in a few lines of code. However, in practice, things typically 
get a little more complicated: you might want to use complicated covariance functions and mean functions, learn good values 
for hyperparameters, use non-Gaussian likelihood functions (rendering exact inference intractable), use approximate inference 
algorithms, or combinations of many or all of the above. 



List of Functionality
------------------------
+-------------+-------------------+------------+------------------------+---------------+---------------------+
|Functionality| Kernel            | Mean       | Likelihood             | Inference     | Optimizer           |
+=============+===================+============+========================+===============+=====================+
| Simple      | Constant          |  Constant  | Gaussian               | Exact         | BFGS                |
|             +-------------------+------------+------------------------+---------------+---------------------+
|             | Linear            |  Linear    |Cumulative Gaussian(Erf)| EP            | CG                  |
|             +-------------------+------------+------------------------+---------------+---------------------+
|             | Linear with ARD   |  One       | Laplace                | Laplace       | SCG                 |
|             +-------------------+------------+------------------------+---------------+---------------------+
|             | Matern            |  Zero      | Logistic               |               | Rasmussen's Minimize|
|             +-------------------+------------+------------------------+               +---------------------+
|             | Periodic          |            |                        |               |                     |
|             +-------------------+            |                        |               |                     |  
|             | Polynomial        |            |                        |               |                     |
|             +-------------------+            |                        |               |                     |
|             | Piecewise Poly    |            |                        |               |                     |
|             +-------------------+            |                        |               |                     |
|             | RBF               |            |                        |               |                     |
|             +-------------------+            |                        |               |                     |
|             | RBF with ARD      |            |                        |               |                     |
|             +-------------------+            |                        |               |                     |
|             | RBF unit          |            |                        |               |                     |
|             +-------------------+            |                        |               |                     |
|             | RQ                |            |                        |               |                     |
|             +-------------------+            |                        |               |                     |
|             | RQ  with ARD      |            |                        |               |                     |
|             +-------------------+            |                        |               |                     |
|             | Noise             |            |                        |               |                     |
+-------------+-------------------+------------+------------------------+---------------+---------------------+
| Composition | Sum               |  Sum       |                        |               |                     |
|             +-------------------+------------+                        |               |                     |
|             | Product           |  Product   |                        |               |                     |
|             +-------------------+------------+                        |               |                     |
|             | Scale             |  Scale     |                        |               |                     |
|             +-------------------+------------+                        |               |                     |
|             |                   |  Power     |                        |               |                     |
+-------------+-------------------+------------+------------------------+---------------+---------------------+
| Sparse GP   |                   |            |                        | FITC EP       |                     |
|             |                   |            |                        +---------------+                     |
|             |                   |            |                        | FITC Laplace  |                     |
+-------------+-------------------+------------+------------------------+---------------+---------------------+
| Structured  | Diffusion         |            |                        |               |                     |
|             +-------------------+            |                        |               |                     |
|             | VN Diffusion      |            |                        |               |                     |
|             +-------------------+            |                        |               |                     |
|             |Pseudo Inv Laplace |            |                        |               |                     |
|             +-------------------+            |                        |               |                     |
|             |Regularized Laplace|            |                        |               |                     |
+-------------+-------------------+------------+------------------------+---------------+---------------------+
| Other       | Customized        | Customized |                        |               |                     |
|             +-------------------+------------+------------------------+               |                     |
|             |Precomputed Matrix |            |                        |               |                     |
+-------------+-------------------+------------+------------------------+---------------+---------------------+

pyGPs also provide cross-validation and some built-in evalation methods.
