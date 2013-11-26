'''
This file is NOT a python script
Its a text document to show how to use kernels.
I use .py only to highlight keywords, I intended to put this documentation as a webpage 
'''


=================================================
OPTIMIZATION METHODS 
=================================================
# Optimizer is initialized with following parameters:
model.setOptimizer(method, num_restarts=None, min_threshold=None, meanRange=None, covRange=None, likRange=None)

-> method 
Optimizatin methods. Possible values are:
"Minimize"   # minimize by Carl Rasmussen
"CG"         # conjugent gradient
"BFGS"       # quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)
"SCG"        # scaled conjugent gradient (faster than CG) 

-> num_restarts
Set if you want to run mulitiple times of optimization with different initial guess. 
It specifys the maximum number of runs/restarts/trials.

-> min_threshold
Set if you want to run mulitiple times of optimization with different initial guess. 
It specifys the threshold of objective function value. Stop optimization when this value is reached.

-> meanRange
The range of initial guess for mean hyperparameters.
e.g. meanRange = [(-2,2), (-5,5), (0,1)]
- Each tuple specifys the range (low, high) of this hyperparameter,
- This is only the range of initial guess, 
  during optimization process, optimal hyperparameters may go out of this range.
- (-5,5) for each hyperparameter by default.

-> covRange
The range of initial guess for kernel hyperparameters.

-> likRange
The range of initial guess for liklihood hyperparameters.

