GPs & Functionality
========================
Gaussian Processes (GPs) can conveniently be used for Bayesian supervised learning, such as regression and classification. 
In its simplest form, GP inference can be implemented in a few lines of code. However, in practice, things typically 
get a little more complicated: you might want to use expressive covariance and mean functions, learn good values 
for hyperparameters, use non-Gaussian likelihood functions (rendering exact inference intractable), use approximate inference 
algorithms, or combinations of many or all of the above. 

A comprehensive introduction to Gaussian Processes for Machine Learning is provided in the `GPML`_ book by Rasmussen and Williams, 2006.

.. _GPML: http://www.gaussianprocess.org/gpml
