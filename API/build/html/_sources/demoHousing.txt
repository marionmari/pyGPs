Regression on UCI Housing data
------------------------------

Boston Housing is a fairly standard dataset used for testing regression problems. It contains 506 data points with 12 numeric attributes, and one binary 
categorical attribute.  The goal is to determine median home values, based on various census attributes. This dataset is available at the `UCI 
Repository`_. 

The demo follows that in [1]_.  The data set was preprocessed as follows: each continuous feature was transformed to zero mean and
unit variance (The categorical variable was dropped).  The data was partitioned into :math:`481` points for training and :math:`25` points for testing.

The mean function used was :func:`src.Core.means.meanZero` and the covariance (using the :func:`src.Core.kernels.covSum` function) was a composite of
:func:`src.Core.kernels.covSEiso` and :func:`src.Core.kernels.covNoise`.  The initial values of the hyperparameters were selected randomly from a zero-mean, 
unit-variance normal distribtion.  The actual values were: :math:`[ -0.75, \; 0.59, \; -0.45 ]`. The initial likelihood hyperparameter
was :math:`-2.30`.  The regression started with initial negative log marginal likelihood of :math:` 752.46`.  Note the initial zero mean and the 
variance that is uniform over the test set. ::

    model = pyGPs.GPR()
    model.optimize(x,y)
    ym, ys2, fm, fs2, lp = model.predict(xs)
    xa  = np.concatenate((data[:,:4],data[:,5:-1]),axis=1)
    xa = (xa - np.mean(xa,axis=0))/(np.std(xa,axis=0)+1.e-16)
    ya, ys2a, fma, fs2a, lpa = model.predict(xa)

.. figure:: _images/demoH1.png
   :align: center
   :width: 400pt
   :height: 300pt

After hyperparameter optimization, the covariance hyperparameters were :math:`[ 1.17, \;  0.45, \; -1.41 ]` and the likelihood 
hyperparameter was :math:`-2.27`.  The final negative log marginal likelihood (optimized) was  :math:`214.46`.

.. figure:: _images/demoH2.png
   :align: center
   :width: 400pt
   :height: 300pt

.. _UCI Repository: http://archive.ics.uci.edu/ml/datasets/Housing

.. [1] T. Suttorp and C. Igel, Approximation of Gaussian process regression models after training. In M. Verleysen (Hrsg.), Proceedings of the 16th European Symposium on Artificial Neural Networks (ESANN 2008) , pp. 427–432 (2008).
