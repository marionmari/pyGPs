"""

__author__ = ['christiaanleysen']

This example divides a set of time-series into two clusters of the most similar time-series using the general
model learn over a set of time-series.

Find more information in the following paper:

"Energy consumption profiling using Gaussian Processes",
Christiaan Leysen*, Mathias Verbeke†, Pierre Dagnely†, Wannes Meert*
*Dept. Computer Science, KU Leuven, Belgium
†Data Innovation Team, Sirris, Belgium
https://lirias.kuleuven.be/bitstream/123456789/550688/1/conf2.pdf
"""
import pyGPs.Core.gp as pyGPs
import scipy
import numpy as np
import timeit
import logging


logger = logging.getLogger("pyGPs.clustering")
ValuesY = []


def gp_likelihood_independent(hyperparams, model, xs, ys, der=False):
    """
    find the aggregated likelihoods of the Gaussian process regression
    Parameters:
    -----------

    hyperparams: hyperparameters for the Gaussian process regression that are used used.
    model: GPR model
    xs: the list of featureset
    ys: the list of valueset
    der: boolean to also minimize the derivatives of the hyperparameters


    Returns:
    --------
    the accumulated likelihood of the Gaussian process regression
    """
    global ValuesY

    # set the hyperparameters
    model.covfunc.hyp = hyperparams.tolist()
    likelihoodList = []

    # accumulate all negative log marginal likelihood (model.nlZ) and the derivative (model.dnlZ)
    all_nlZ = 0
    all_dnlZ = pyGPs.inf.dnlZStruct(model.meanfunc, model.covfunc, model.likfunc)

    for x, y in zip(xs, ys):
        model.setData(x, y)
        if der:
            this_nlZ, this_dnlZ, post = model.getPosterior(der=der)
            all_nlZ += this_nlZ
            all_dnlZ = all_dnlZ.accumulateDnlZ(this_dnlZ)
            likelihoodList.append(this_nlZ)
        else:
            this_nlZ, post = model.getPosterior(der=der)
            all_nlZ += this_nlZ
            likelihoodList.append(this_nlZ)

    # calculate weighted means by making use of the relative likelihoods.
    likelihoodList = [abs(i/np.sum(abs(i) for i in likelihoodList)) for i in likelihoodList]
    ValuesY = [i*j.tolist() for i,j in zip(ys,likelihoodList)]
    ValuesY = np.array([sum(i) for i in zip(*ValuesY)])


    returnValue = all_nlZ
    if der:
        returnValue = all_nlZ+np.sum(all_dnlZ.cov)+np.sum(all_dnlZ.mean)
    return returnValue



def optimizeHyperparameters(initialHyperParameters, model, xs, ys, bounds=None, method='BFGS'):
    """
    Optimize the hyperparameters of the general Gaussian process regression
    Parameters:
    -----------

    initialHyperparameters: initial hyper parameters used.
    model: GPR model
    xs: the list of featureset
    ys: the list of valueset
    bounds: the bounds needed for the minimize method (if needed).
    method: the minimize method that is employed e.g. BFGS

    Returns:
    --------
    the optimal hyperparameters and the model
    """
    global ValuesY
    ValuesY = []
    if bounds is None:
        bounds = []

    logger.info('optimizing Hyperparameters...')
    start = timeit.default_timer()
    result = scipy.optimize.minimize(gp_likelihood_independent, initialHyperParameters, args=(model,xs,ys),bounds=bounds,method=method) #powell gaat lang
    stop = timeit.default_timer()
    logger.info("minimization time:", stop - start)

    hyperparams = result.x
    model.covfunc.hyp = hyperparams.tolist()
    model.getPosterior(xs[0], ValuesY)

    return hyperparams, model
