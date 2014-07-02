Multi-class Classification
===========================

GPMC is NOT based on multi-class Laplace approximation.
It works as a one vs. one classification wrapper. 
In other words, GPMC trains a GPC model for each pair of two classes,
and uses a majority voting scheme over all results to determine the final class.
The method only returns the predictive class with highest rating; 
no other values (such as variance) are returned. 

Lets see a practical example to classify the 10 (0,1,2,...9) hand-written digits 
in the USPS digits dataset.


Load data
--------------------
The USPS digits data were gathered at the Center of Excellence in Document Analysis and Recognition (CEDAR) at SUNY Buffalo, as part of a project sponsored by the US Postal Service. The dataset is described in [1]_. ::

	data = loadmat('data_for_demo/usps_resampled.mat')
	x = data['train_patterns'].T   # train patterns
	y = data['train_labels'].T     # train labels
	xs = data['test_patterns'].T   # test patterns
	ys = data['test_labels'].T     # test labels 

To be used in GPMC, labels should start from 0 to k (k = number of classes). 


GPMC example
---------------------
State model with 10-class classification problem: ::

	model = gp.GPMC(10)

Pass data to model: ::

	model.setData(x,y)

Train default GPC model for each binary classification problem, 
and decide label for test patterns of hand-writen digits.
The return value *predictive_vote[i,j]* is the probability of being class *j* for test pattern *i*. ::

	predictive_vote = model.trainAndPredict(xs)
	predictive_class = np.argmax(predictive_vote, axis=1)

Just like we did for GP classification, 
you can use specific settings (other than default) for all binary classificiation problem for example by: ::

	m = mean.Zero()
	k = cov.RBF()
	model.setPrior(mean=m,kernel=k)
	model.useInference("Laplace")

For more information on how to use non-default settings see `demo_GPC`_ and `demo_GPR`_.  


.. _demo_GPC: GPC.html 
.. _demo_GPR: GPR.html 


Beside *trainAndPredict(xs)*, 
there is also an option to perform prediction without hyperparameter optimization: ::

    model.fitAndPredict(xs)

.. [1] A Database for Handwritten Text Recognition Research, J. J. Hull, IEEE PAMI 16(5) 550-554, 1994.
