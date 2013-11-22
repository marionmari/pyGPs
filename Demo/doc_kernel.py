# (If you will optimize anyway, just leave parameters to be default)

# parameters are passed in logorithms

default init need dimension info. (either D or init)
# m = mean.Linear( D=x.shape[1] )
cov.RBFard


No signal variance(scalar term) in kernel itself:
(use *)
cov.LIN
cov.LINard



#  scarlar is also hyperparameter

"""

# TODO  add demo_kernel 
# @see demo_kernel  for all default settings
# @see demo_kernel  for how to use kernel(mean) compositions and how to set hyperparameters
# @add demo_kernel  for why using logarithm of ell and sigma (to ensure positive)
#  discription for covPre
"""

MEAN:
----------------------------
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
	d -> order of polynomial
	sigma -> signal deviation

# Squared Exponential kernel with isotropic distance measure
RBF( log_ell=-1., log_sigma=0.)
	ell -> characteristic length scale
	sigma -> signal deviation

# Squared Exponential kernel with isotropic distance measure with unit magnitude
# i.e signal variance is always 1
RBFunit( log_ell=-1. )         
	ell -> characteristic length scale

# Squared Exponential kernel with Automatic Relevance Determination
RBFard( log_ell_list=[0.5 for i in xrange(D)], log_sigma=0.)
	log_ell_list -> logarithm of characteristic length scale for each dimension
	sigma -> signal deviation

# Constant kernel
Const( log_sigma=0. )
	sigma -> signal deviation

# Linear kernel
LIN()

# Linear kernel with Automatic Relevance Determination
LINard( log_ell_list=[0.5 for i in xrange(D)] )
	log_ell_list -> logarithm of ARD parameters for each dimension

# Matern kernel with nu = d/2 and isotropic distance measure
# For d=1 the function is also known as the exponential covariance function or the 
# Ornstein-Uhlenbeck covariance in 1d.
Matern( log_ell=-1., log_d=0., log_sigma=0. )
	d -> 2 times nu
	ell -> characteristic length scale
	sigma -> signal deviation	

# Stationary kernel for a smooth periodic function
Periodic( log_ell=-1, log_p=0., log_sigma=0. )
	ell -> characteristic length scale
	p -> period
	sigma -> signal deviation

# Independent covariance function, ie "white noise"
# !!! Not used since noise is now added in liklihood!
Noise( log_sigma=0. )
	sigma -> signal deviation








