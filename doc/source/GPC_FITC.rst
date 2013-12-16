Sparse Classification
========================
The code shown in this tutorial can be obtained by running */pyGP_OO/Demo/demo_GPC_FITC.py*
This demo is more or less repeated of the demo of FITC regression.

First example -> default inducing points
-------------------
First load the same data as in the GPC demo.

**[Theory]**
In case the number of training inputs :math:`x` exceeds a few hundred, approximate inference using Laplacian Approximation or Expectation Propagation takes too long. As in regression, we offer the FITC approximation 
based on a low-rank plus diagonal approximation to the exact covariance to deal with these cases. The general idea is to use inducing points 
:math:`u` and to base the computations on cross-covariances between training, test and inducing points only.

Okay, now the model is FITC classificiation ::

	model = gp.GPC_FITC()  

The difference betwwen the usage of basic gp is that we will have to specify inducing points.
In the first example here, we'll introduce you how to use default settings.

The default inducing points is a grid(hypercube in higher dimension), where each dimension has 5 values in same step between min and max value of data by default. In order to let the model know the dimension of input data, we HAVE TO set data first. ::

    model.setData(x, y)

This number of value per axis for default inducing points can also be changed ::

    model.setData(x, y, value_per_axis=10)

Then the regular process follows: ::

	model.train()           
	model.predict(z, ys=np.ones((z.shape[0],1))) 
	model.plot(x1,x2,t1,t2)

.. figure:: _images/d4_1.png
   :height: 600 px
   :width: 800 px
   :align: center
   :scale: 70 %

The equispaced default inducing points :math:`u` that are shown in the figure as black circles.


Second example -> user-defined inducing points
-----------------------------

Alternatively, a random subset of the training points can be used as inducing points. Moreover, there are planty of methods to set these inducing points.
So in the second example lets use a user-defined set of inducing points. ::

	u1,u2 = np.meshgrid(np.linspace(-2,2,5),np.linspace(-2,2,5))
	u = np.array(zip(np.reshape(u2,(np.prod(u2.shape),)),np.reshape(u1,(np.prod(u1.shape),)))) 

Here, we also use a grid euqually spaced, but without the values on the margin of the grid.(i.e. shrinking the grid) Then, we can just pass :math:`u` when specifying prior. ::

	m = mean.Zero()
	k = cov.RBFard(log_ell_list=[0.05,0.17], log_sigma=1.)
	model.setPrior(mean=m, kernel=k, inducing_points=u) 

The predicting results for this inducing points are shown below

.. figure:: _images/d4_2.png
   :height: 600 px
   :width: 800 px
   :align: center
   :scale: 70 %


A bit more things you can do
------------------------------
As GPC, it is possible to use FITC Laplacian Approximation by ::

    model.useLaplace_FITC() 

