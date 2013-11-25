'''
This file is NOT a python script
Its a text document to show how to use kernels.
I use .py only to highlight keywords, I intended to put this documentation as a webpage 
'''


=================================================
SIMPLE KERNELS 
=================================================
# You may already seen, we can specify a kernel function like this:
# (similar for mean fucntions as well)
k = cov.RBF( log_ell=-1., log_sigma=0. )


There are several points need to be noticed:

1. Most parameters are initilized in their logorithms. 
This is because we need to make sure they are positive during optimization.
e.g. Here length scale and signal variance should always be positive.
@SEE Section: ALL KERNELS AND MEANS & DEFAULT PARAMETERS

2. Most kernel functions have a scalar in front, namely signal variance(set by log_sigma)

3. If you will do optimization later anyway, you can just leave parameters to be default











=================================================
SOME SPECIAL CASES
=================================================
1. 
For some kernels/means, number of hyperparameters depends on the dimension of input data

# You can either enter the dimension, which use default values:
m = mean.Linear( D=x.shape[1] )

# or you can initialze with the exact hyperparameters,
# you should enter as a list, one element for each dimension
m = mean.Linear( alpha_list=[0.2, 0.4, 0.3] )

# These "hyp-dim-dependent" functions are:
mean.Linear
cov.RBFard
cov.LINard


2. 
For linear kernel, there is NO signal variance(scalar) in front of the function.
# If you want to add a scalar for it, you can use:
k = 0.5 * cov.LIN()
# If you also want to add a bias term:
k = 0.5 * cov.LIN() + cov.Const(c=1.)

Note 0.5 will also be treated as a hyperparameter.
This also applies in cov.LINard.


3. 
For cov.RBFunit(), its signal variance is always 1(because of unit magnitude)
so this function do not have a hyperparameter of "signal variance".


4.
cov.Poly() has three parameters, where
hyperparameters are:
	c -> inhomogeneous offset
	sigma -> signal deviation 
	
however, 
	d -> order of polynomial 
	will be treated as normal parameter, i.e. will not be trained


5.
Explicitly set cov.Noise is not necessary, 
because noise are already added in liklihood.









=================================================
COMPOSITE KERNELS 
=================================================
# Adding and muliplying kernels is really simple:
k = cov.Periodic() * cov.RBF()
k = 0.5*cov.LIN() + cov.Periodic()

Except linear kernel, all kernel functions have a scalar(signal variance) as hyperparameter.
Therefore, the only explict scalar is added to cov.LIN()
 
# Beside +/*, There is even a power operator for mean functions:
m = ( mean.One()+mean.Linear(alpha_list=[0.2]) )**2










=================================================
FITC APPROXIMATION
=================================================













=================================================
PRECOMPUTED KERNEL MATRIX
=================================================
In certain cases, you may have a precomputed kernel matrix,
but its non-trivial to write down the exact formula of kernel functions.

# Then you can specify your kernel like this:
k = cov.Pre(M1, M2)

M1 and M2 are your precomputed kernel matrix,

where,
M1 (shape: train+1, test)  
    -> cross covariances matrix (train by test) 
    -> last row is self covariances (diagonal of test by test)
M2 (shape: train, train)
    -> training set covariance matrix (train by train)  


# A precomputed kernel can also be composited with other kernels.
k = 0.5*cov.Pre(M1, M2) + cov.RBF()

Note: Similar to cov.LIN(), you need to explictly add scalar for cov.Pre()










=================================================
LIST OF ALL KERNELS/MEANS & DEFAULT PARAMETERS
=================================================
MEAN:
------------------------------
Zero() 

One()

Const( c=5. )     
	c -> constant

Linear( alpha_list=[0.5 for i in xrange(D)] )    
	alpha_list -> alpha for each dimension


KERNEL:
------------------------------
# Polynomial kernel
Poly( log_c=0., log_d=np.log(2), log_sigma=0. )
	c -> inhomogeneous offset
	d -> order of polynomial (treated not as hyperparameter, i.e. will not be trained)
	sigma -> signal deviation 
	hyp = [ log_c, log_sigma]

# Squared Exponential kernel with isotropic distance measure
RBF( log_ell=-1., log_sigma=0.)
	ell -> characteristic length scale
	sigma -> signal deviation
	hyp = [log_ell, log_sigma]

# Squared Exponential kernel with isotropic distance measure with unit magnitude
# i.e signal variance is always 1
RBFunit( log_ell=-1. )         
	ell -> characteristic length scale
	hyp = [log_ell]

# Squared Exponential kernel with Automatic Relevance Determination
RBFard( log_ell_list=[0.5 for i in xrange(D)], log_sigma=0.)
	log_ell_list -> logarithm of characteristic length scale for each dimension
	sigma -> signal deviation
	hyp = [log_ell for each dimension, log_sigma]

# Constant kernel
Const( log_sigma=0. )
	sigma -> signal deviation
	hyp = [log_sigma]

# Linear kernel
LIN()

# Linear kernel with Automatic Relevance Determination
LINard( log_ell_list=[0.5 for i in xrange(D)] )
	log_ell_list -> logarithm of ARD parameters for each dimension
	hyp = [log_ell for each dimension]

# Matern kernel with nu = d/2 and isotropic distance measure
# For d=1 the function is also known as the exponential covariance function or the 
# Ornstein-Uhlenbeck covariance in 1d.
Matern( log_ell=-1., log_d=0., log_sigma=0. )
	d -> 2 times nu
	ell -> characteristic length scale
	sigma -> signal deviation
	hyp = [ log_ell, log_sigma, log_d ]	

# Stationary kernel for a smooth periodic function
Periodic( log_ell=-1, log_p=0., log_sigma=0. )
	ell -> characteristic length scale
	p -> period
	sigma -> signal deviation
	hyp = [ log_ell, log_p, log_sigma]

# Independent covariance function, ie "white noise"
# !!! Not used since noise is now added in liklihood!
Noise( log_sigma=0. )
	sigma -> signal deviation
	hyp = [log_sigma]








