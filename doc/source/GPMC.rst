Multi-class Classification
=========================

GPMC is NOT based on multi-class Laplace Approximation.
It works as a one vs. one classification wrapper. 
In other words, GPMC trains GPC model for each combinations of two classes,
and uses voting scheme over all results to determine the final class.
The method only returns the predictive class with highest rating, 
but no other values (such as variance) returned. 

Lets see a practical example to classify 10(0~9) hand-writen digits,
using USPS digits dataset.


Load data
--------------------
The USPS digits data were gathered at the Center of Excellence in Document Analysis and Recognition (CEDAR) at SUNY Buffalo, as part of a project sponsored by the US Postal Service. The dataset is described in [1]_. ::

	data = loadmat('data_for_demo/usps_resampled.mat')
	x = data['train_patterns'].T   # train patterns
	y = data['train_labels'].T     # train labels
	xs = data['test_patterns'].T   # test patterns
	ys = data['test_labels'].T     # test labels 

To be used in GPMC, labels should start from 0 to n(number of classes) ::


GPMC example
---------------------
State model with 10-classes classification problem. ::

	model = gp.GPMC(10)

Pass data to model. ::

	model.setData(x,y)

Train default GPC model (see demo_GPC) for each binary classification problem, 
and decide label for test patterns of hand-writen digits.
The return value *prdictive_vote[i,j]* is the probability of being class j for test pattern i. ::

	predictive_vote = model.trainAndPredict(xs)
	predictive_class = np.argmax(predictive_vote, axis=1)

Just like we did for GP classification, 
you can use specify settings(other than default) for each binary classificiation problem for example by: ::

	m = mean.Zero()
	k = cov.RBF()
	model.setPrior(mean=m,kernel=k)
	model.useLaplace()
	model.setOptimizer("SCG", num_restarts=20)


Beside *trainAndPredict(xs)*, 
there is also an option to predict without optimization. ::

    model.fitAndPredict(xs)


.. [1] A Database for Handwritten Text Recognition Research, J. J. Hull, IEEE PAMI 16(5) 550-554, 1994.