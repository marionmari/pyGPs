K-fold Cross-Validation
==================================

In this demo, we'll show you the typical process of using GP for machine learning from loading data, preprocessing, training,  predicting to validation and evaluation.

Load data
--------------------
We use the ionosphere dataset from UCI machine learning repository. ::

	data_source = "data_for_demo/ionosphere.data.txt"
	x = []
	y = []
	with open(data_source) as f:
	    for index,line in enumerate(f):
	        feature = line.split(',')
	        attr = feature[:-1]
	        attr = [float(i) for i in attr]
	        target = [feature[-1]]       
	        x.append(attr)
	        y.append(target)
	x = np.array(x)
	y = np.array(y)

Preprocessing
-------------
Then, we do some data cleaning. Here we deal with label in ionosphere data, change "b" to"-1", and "g" to "+1" ::

	n,D = x.shape
	for i in xrange(n):
	    if y[i,0][0] == 'g':
	        y[i,0] = 1
	    else:
	        y[i,0] = -1
	y = np.int8(y)

Cross Validation
--------------
Now, lets focus on the use of cross-validation. Lets see the complete process by this example. ::

	K = 10             # number of fold
	ACC = []           # accuracy 
	RMSE = []          # root-mean-square error
	cv_run = 0

	for x_train, x_test, y_train, y_test in valid.k_fold_validation(x, y, K):
	    print 'Run:', cv_run
	    # This is a binary classification problem
	    model = gp.GPC()
	    # Since no prior knowldege, leave everything default 
	    model.train(x_train, y_train)
	    # Predit 
	    ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys=y_test)

	    # ymu for classification is a continuous value over -1 to +1
	    # If you want predicting result to either one of the classes, take a sign of ymu.
	    ymu_class = np.sign(ymu)

	    # Evluation
	    acc = valid.ACC(ymu_class, y_test)
	    print '   accuracy =', round(acc,2) 
	    rmse = valid.RMSE(ymu_class, y_test)
	    print '   rmse =', round(rmse,2)
	    ACC.append(acc)
	    RMSE.append(rmse)

	    # Toward next run
	    cv_run += 1   
 


Evaluation measures
---------------
We implemented some classical evaluation measures. 
    - RMSE - Root mean squared error
    - ACC - Classification accuracy
    - Prec - Precision for class +1
    - Recall - Recall for class +1
    - NLPD - Negative log predictive density in transformed observation space