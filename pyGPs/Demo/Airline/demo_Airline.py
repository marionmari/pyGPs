from __future__ import print_function
from builtins import range
#================================================================================
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [dan dot marthaler at gmail dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]
#
#    This file is part of pyGPs.
#    The software package is released under the BSD 2-Clause (FreeBSD) License.
#
#    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 18/02/2014
#================================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import io as sio

import pyGPs

if __name__ == '__main__':
    data = sio.loadmat('airlinedata.mat')

    x = np.atleast_2d(data['xtrain'])
    y = np.atleast_2d(data['ytrain'])
    xt = np.atleast_2d(data['xtest'])
    yt = np.atleast_2d(data['ytest'])

    # To get interpolation too
    #xt = np.concatenate((x,xt))
    #yt = np.concatenate((y,yt))

    # Set some parameters
    Q = 10

    model = pyGPs.GPR()           # start from a new model

    # Specify non-default mean and covariance functions
    # @SEE doc_kernel_mean for documentation of all kernels/means
    m = pyGPs.mean.Zero()

    for _ in range(10):
      k = pyGPs.cov.SM(Q)
      k.initSMhypers(x, y)
      model.setPrior(kernel=k)

      # Noise std. deviation
      sn = 0.1

      model.setNoise(log_sigma=np.log(sn))
      # Instead of getPosterior(), which only fits data using given hyperparameters,
      # optimize() will optimize hyperparamters based on marginal likelihood
      # the deafult mean will be adapted to the average value of the training labels..
      # ..if you do not specify mean function by your own.
      model.optimize(x, y)
      print(_, model.nlZ)
      model.predict(xt)
      model.plot()

    '''
    print 'Optimized negative log marginal likelihood:', round(model.nlZ, 3)
    # Predict test data
    # output mean(ymu)/variance(ys2), latent mean(fmu)/variance(fs2), and log
    # predictive probabilities(lp)
    ym, ys2, fmu, fs2, lp = model.predict(xt)

    # Plot the stuff
    plt.plot(x, y, 'b', label=u'Training Data')
    plt.plot(xt, yt, 'k', label=u'Test Data')
    plt.plot(xt, ym, 'r', label=u'SM Prediction')
    fillx = np.concatenate([np.array(xt.ravel()).ravel(),
                            np.array(xt.ravel()).ravel()[::-1]])
    filly = np.concatenate([(np.array(ym.ravel()).ravel() - 1.9600 *
                             np.array(ys2.ravel()).ravel()),
                            (np.array(ym.ravel()).ravel() + 1.9600 *
                             np.array(ys2.ravel()).ravel())[::-1]])
    plt.fill(fillx, filly, alpha=.5, fc='0.5', ec='None',
             label='95% confidence interval')

    plt.show()
   '''
