import numpy as np
def initialize_hyperparameters(x):
    '''Initialize the hyperparameters utilizing the initilization methods
       discussed in Ryan Turner's Thesis: Gaussian Processes for State
       Space Models and Changepoint Detection (pp 62-63)
    '''
    c1     = 1.0
    c2     = 1.0
    delta  = np.median(np.abs(np.diff(x,axis=0)))
    R      = np.percentile(x,95)
    mu     = 0.5*np.log(c2 * R * delta)
    sig2   = (0.25*np.log(c1 * R / delta))**2
    # Do a bit of error handling
    if np.isinf(mu) or np.isinf(sig2):
        theta0 = np.median(x)
    else:
        theta0 = np.random.normal(mu,sig2)
    if np.isnan(theta0):
        return np.median(x)
    return theta0
