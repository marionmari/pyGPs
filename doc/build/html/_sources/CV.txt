K-fold Cross-Validation
==================================

In this demo, we'll show you the typical process of using GP for machine learning from loading data, preprocessing, training,  predicting to validation and evaluation.

Load data
--------------------
We use the ionosphere dataset [1]_ from Johns Hopkins University Ionosphere database. 
It is available in UCI machine learning repository. ::
Then we need to do some data cleaning. Here we deal with label in ionosphere data, change "b" to"-1", and "g" to "+1". These preprocessing implementation are availabe in the source code.


Cross Validation
--------------
Now, lets focus on the use of cross-validation. ::

	K = 10             # number of fold
	for x_train, x_test, y_train, y_test in valid.k_fold_validation(x, y, K):
	    # This is a binary classification problem
	    model = gp.GPC()
	    # Since no prior knowldege, leave everything default 
	    model.train(x_train, y_train)
	    # Predition 
	    ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys=y_test)
	    # ymu for classification is a continuous value over -1 to +1
	    # If you want predicting result to either one of the classes, take a sign of ymu.
	    ymu_class = np.sign(ymu)
	    # Evluation
	    acc = valid.ACC(ymu_class, y_test)       # accuracy 
	    rmse = valid.RMSE(ymu_class, y_test)     # root-mean-square error


Evaluation measures
---------------
We implemented some classical evaluation measures. 
    - RMSE - root mean squared error
    - ACC - classification/regression accuracy
    - Prec - classification precision for class +1
    - Recall - classification recall for class +1
    - NLPD - negative log predictive density in transformed observation space




.. [1] Sigillito, V. G., Wing, S. P., Hutton, L. V., \& Baker, K. B. (1989). Classification of radar returns from the ionosphere using neural networks. Johns Hopkins APL Technical Digest, 10, 262-266. 
