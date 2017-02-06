"""

__author__ = ['christiaanleysen', 'wannesmeert']

This example divides a set of time-series into two clusters of the most similar time-series using the general
model learn over a set of time-series.

Find more information in the following paper:

"Energy consumption profiling using Gaussian Processes",
Christiaan Leysen*, Mathias Verbeke†, Pierre Dagnely†, Wannes Meert*
*Dept. Computer Science, KU Leuven, Belgium
†Data Innovation Team, Sirris, Belgium
https://lirias.kuleuven.be/bitstream/123456789/550688/1/conf2.pdf
"""
import sys
import pyGPs
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from pyGPs.Demo.Clustering import pyGP_extension as gpe
import numpy as np
import matplotlib.pyplot as plt
import math
import logging
import random
from collections import namedtuple

logger = logging.getLogger("pyGPs.clustering")


def calculate_rmse_gp(vector_x, vector_y, weighted=True, plot=False, context=None, optimization_params=None,
                      signed=False, sample=None):
    """Calculate the root mean squared error.

    :param vector_x: timestamps of the timeseries
    :param vector_y: valueSet of the timeseries
    :param weighted: weight RMSE wrt variance of prediction
    :param plot: plot the expected function
    :param context: (internal)
    :param optimization_params:
    :param signed: Add a sign to RMSE based on whether the prediction is on average higher or lower than the prediction
    :param sample: Learn from sample of the data (int for min number, float for fraction, list for inidices)
    :returns: list(idx,rmse), hyperparams, model
    """
    if optimization_params is None:
        optimization_params = {}
    # setX = [preprocessing.scale(element )for element in vectorX]
    # setY = preprocessing.scale(vector_y, axis=1)

    vector_y_train = vector_y
    vector_x_train = vector_x
    if sample:
        if type(sample) == float:
            logger.debug("Sample series for training (ratio)")
            vector_y_train = []
            vector_x_train = []
            for idx in random.sample(range(len(vector_y)), k=int(len(vector_y) * sample)):
                vector_y_train.append(vector_y[idx])
                vector_x_train.append(vector_x[idx])
        elif type(sample) == int:
            logger.debug("Sample series for training (number)")
            if len(vector_y) <= sample:
                vector_y_train = vector_y
                vector_x_train = vector_x
            else:
                vector_y_train = []
                vector_x_train = []
                for idx in random.sample(range(len(vector_y)), k=sample):
                    vector_y_train.append(vector_y[idx])
                    vector_x_train.append(vector_x[idx])
        elif type(sample) == list:
            logger.debug("Sample series for training (indices)")
            vector_y_train = []
            vector_x_train = []
            for idx in sample:
                vector_y_train.append(vector_y[idx])
                vector_x_train.append(vector_x[idx])

    model = pyGPs.GPR()  # specify model (GP regression)
    k = pyGPs.cov.Linear() + pyGPs.cov.RBF()  # hyperparams will be set with optimizeHyperparameters method
    model.setPrior(kernel=k)

    hyperparams, model2 = gpe.optimizeHyperparameters(
        optimization_params.get("initialHyperParameters", [0.0000001, 0.0000001, 0.0000001]),
        model, vector_x_train, vector_y_train,
        bounds=optimization_params.get("bounds", [(None, 5), (None, 5), (None, 5)]),
        method=optimization_params.get("method", 'L-BFGS-B'))
    logger.info('Hyperparameters used: {}'.format(hyperparams))
    # mean (y_pred) variance (ys2), latent mean (fmu) variance (fs2), log predictive prob (lp)
    y_pred, ys2, fm, fs2, lp = model2.predict(vector_x[0])
    last_vector_x = vector_x[0]

    rmse_data = []
    for i in range(len(vector_y)):
        if not np.all(np.equal(last_vector_x, vector_x[i])):
            logger.debug("Recomputing prediction")
            y_pred, ys2, fm, fs2, lp = model2.predict(vector_x[i])
            last_vector_x = vector_x[i]
        if weighted:
            rmse = math.sqrt(mean_squared_error(vector_y[i], y_pred, (np.max(ys2) - ys2)) / np.max(ys2))
        else:
            rmse = math.sqrt(mean_squared_error(vector_y[i], y_pred))
        if signed:
            if np.mean(vector_y[i] - y_pred) < 0:
                rmse = -rmse
        rmse_data.append((i, rmse))

    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 2))
        xs = vector_x[0]
        ym = y_pred
        xss = np.reshape(xs, (xs.shape[0],))
        ymm = np.reshape(ym, (ym.shape[0],))
        ys22 = np.reshape(ys2, (ys2.shape[0],))
        for i in vector_y:
            ax[0].plot(i, color='blue', alpha=0.2)
        ax[0].set_title("Node {}".format(context["cum_depth"]))
        ax[0].fill_between(xss, ymm + 3. * np.sqrt(ys22), ymm - 3. * np.sqrt(ys22),
                           facecolor=[0.7539, 0.89453125, 0.62890625, 1.0], linewidth=0.5)
        ax[0].plot(xss, ym, color='red', label="Prediction")
        ax[0].legend()
        rmse_list = [t[1] for t in rmse_data]
        ax[1].hist(rmse_list, bins=100)
        ax[1].vlines(np.mean(rmse_list), 0, 2, color="red")
        ax[1].set_xlabel("RMSE")
        ax[1].set_ylabel("#")
        # plt.show(block=True)

    return rmse_data, hyperparams, model2


def hierarchical_step(series, split_rmse=None, split_avgrmse=None, split_ratio=None,
                      max_avgrmse=None, min_size=None,
                      weighted=True, signed=False, sample=None,
                      plot=False, context=None, optimization_params=None):
    """
    aux method for the clustering which divides the clusterlist further into clusters using a certain threshold.

    :param series: (labels, values_x, values_y)
    :param split_rmse: Split on this rmse (optional)
    :param split_avgrmse: Split on the average rmse (or split_avgrms*avgrmse if a number is given)
    :param max_avgrmse: mean similarity threshold to divide the clusters, otherwise do not split
    :param min_size: minimum cluster size, otherwise do not split
    :param split_ratio: ratio of timeseries that will be devided into the left and right cluster (optional)
    :param weighted: Weight RMSE based on confidence of GP prediction
    :param plot: Plot the intermediate steps of the algorithm
    :param context: For internal use.
    :param optimization_params: Dict with optional fields initialHyperParameters, bounds, method
    :param signed: Use positive and negative rmse based on whether the prediction is on average lower or higher
        than the actual series
    :param sample: Learn only on a subset of the data (speeds up learning). This can be a number (e.g., 0.8) or a
        list of indices (e.g., [1,3,4,6])
    :returns: (series_left, series_right, model, hyperparams)
    """
    labels, values_x, values_y = series

    list_rmse, hyperparams, model = calculate_rmse_gp(values_x, values_y, weighted=weighted, plot=plot, context=context,
                                                      optimization_params=optimization_params, signed=signed,
                                                      sample=sample)
    sorted_list_rmse = sorted(list_rmse, key=lambda x: x[1])
    mean_rmse = np.mean([t[1] for t in sorted_list_rmse])
    if signed:
        mean_rmse_abs = np.mean([abs(t[1]) for t in sorted_list_rmse])
        logger.info("Split at node, RMSE = [{}, {}/{}, {}]".format(sorted_list_rmse[0][1], mean_rmse,
                                                                   mean_rmse_abs, sorted_list_rmse[-1][1]))
    else:
        mean_rmse_abs = mean_rmse
        logger.info("Split at node, RMSE = [{}, {}, {}]".format(sorted_list_rmse[0][1], mean_rmse,
                                                                sorted_list_rmse[-1][1]))

    if max_avgrmse is not None and mean_rmse_abs < max_avgrmse:
        logger.debug('Avg RMSE too small, stopping')
        return series, None, model, hyperparams
    if min_size is not None and len(values_y) < min_size:
        logger.debug('Cluster size too small, stopping')
        return series, None, model, hyperparams

    # NormalizeValue = sortedListRMSE[-1][1]
    # sortedListRMSE_normalized = [(x[0], x[1] / NormalizeValue) for x in sortedListRMSE][::-1]

    cluster_left_l = []
    cluster_left_x = []
    cluster_left_y = []
    cluster_right_l = []
    cluster_right_x = []
    cluster_right_y = []

    if split_avgrmse is not None:
        mean_rmse = np.mean([t[1] for t in sorted_list_rmse])
        if type(split_avgrmse) in [int, float]:
            mean_rmse *= split_avgrmse
        for i, cur_rmse in sorted_list_rmse:
            if cur_rmse <= mean_rmse:
                cluster_left_l.append(labels[i])
                cluster_left_x.append(values_x[i])
                cluster_left_y.append(values_y[i])
            else:
                cluster_right_l.append(labels[i])
                cluster_right_x.append(values_x[i])
                cluster_right_y.append(values_y[i])
    elif split_ratio is not None:
        # Split based on a ratio between clusters
        cluster_size_length = int(math.ceil(split_ratio * len(sorted_list_rmse)))
        for idx, _ in sorted_list_rmse[-cluster_size_length:]:
            cluster_left_l.append(labels[idx])
            cluster_left_x.append(values_x[idx])
            cluster_left_y.append(values_y[idx])
        for idx, _ in sorted_list_rmse[:len(sorted_list_rmse) - cluster_size_length]:
            cluster_left_l.append(labels[idx])
            cluster_left_x.append(values_x[idx])
            cluster_left_y.append(values_y[idx])
    elif split_rmse is not None:
        # Split based on RMSE
        for i, cur_rmse in sorted_list_rmse:
            if cur_rmse <= split_rmse:
                cluster_left_l.append(labels[i])
                cluster_left_x.append(values_x[i])
                cluster_left_y.append(values_y[i])
            else:
                cluster_right_l.append(labels[i])
                cluster_right_x.append(values_x[i])
                cluster_right_y.append(values_y[i])
    else:
        print("ERROR: either rmse or clusterSize should be set")
        return None

    cluster_left = (cluster_left_l, cluster_left_x, cluster_left_y)
    cluster_right = (cluster_right_l, cluster_right_x, cluster_right_y)

    return cluster_left, cluster_right, model, hyperparams


ClusterNode = namedtuple("ClusterNode", ["left", "right", "model", "hyperparameters", "depth"])
ClusterLeaf = namedtuple("ClusterLeaf", ["series", "model", "hyperparameters", "depth"])


def hierarchical_rec(series, max_depth=None, depth=0, context=None, optimization_params=None, **kwargs):
    """Options are passed to hierarchical_step"""
    logger.info("Hierarchical clustering, level {}".format(depth))
    context["depth"] = depth
    cum_depth = context["cum_depth"]
    cluster_left, cluster_right, model, hyperparams = hierarchical_step(series, context=context,
                                                                        optimization_params=optimization_params,
                                                                        **kwargs)
    if max_depth is not None and depth >= max_depth:
        logger.debug("Max depth reached")
        return ClusterLeaf(series, model, hyperparams, depth)
    if cluster_right is None or not cluster_right[2]:
        logger.debug("Right branch is empty")
        return ClusterLeaf(cluster_left, model, hyperparams, depth)
    if cluster_left is None or not cluster_left[2]:
        logger.debug("Left branch is empty")
        return ClusterLeaf(cluster_right, model, hyperparams, depth)
    context["side"] = "left"
    context["cum_depth"] = cum_depth + " - {}/{}".format(depth, "left")
    left = hierarchical_rec(cluster_left, max_depth=max_depth, depth=depth + 1, context=context,
                            optimization_params=optimization_params, **kwargs)
    context["side"] = "right"
    context["cum_depth"] = cum_depth + " - {}/{}".format(depth, "right")
    right = hierarchical_rec(cluster_right, max_depth=max_depth, depth=depth + 1, context=context,
                             optimization_params=optimization_params, **kwargs)
    return ClusterNode(left, right, model, hyperparams, depth)


def hierarchical(series, max_depth=None, **kwargs):
    """Hierarchical clustering

    :param series: [label, vectorX, vectorY]
    :param max_depth: Max tree depth
    :param kwargs: Args for hierarchical_rec
    :return: (series_left, series_right, model, hyperparams)
    """
    context = {
        "cum_depth": "^"
    }
    return hierarchical_rec(series, max_depth=max_depth, depth=0, context=context, **kwargs)


def print_hierarchical_tree(cluster, cluster_idx=0, output=sys.stdout):
    if type(cluster) == ClusterLeaf:
        labels = [str(l) for l in sorted(cluster.series[0])]
        print("{}Cluster {}: {}".format("  " * cluster.depth, cluster_idx, " ".join(labels)), file=output)
        return cluster_idx + 1
    elif type(cluster) == ClusterNode:
        print("{}Node left".format("  " * cluster.depth), file=output)
        cluster_idx = print_hierarchical_tree(cluster.left, cluster_idx=cluster_idx, output=output)
        print("{}Node right".format("  " * cluster.depth), file=output)
        cluster_idx = print_hierarchical_tree(cluster.right, cluster_idx=cluster_idx, output=output)
        return cluster_idx


def flat_clusters(cluster):
    clusters = []
    if type(cluster) == ClusterLeaf:
        clusters.append(sorted(cluster.series[0]))
    elif type(cluster) == ClusterNode:
        clusters += flat_clusters(cluster.left)
        clusters += flat_clusters(cluster.right)
    return clusters


def visit_leafs(node, fun):
    if type(node) == ClusterLeaf:
        fun(node.series, node.model, node.hyperparameters)
    elif type(node) == ClusterNode:
        visit_leafs(node.left, fun)
        visit_leafs(node.right, fun)


def test():
    series_l = list(range(4))
    series_x = []
    series_y = []
    # Fill the x-values of the time-series with a time between 0 and 20
    for i in range(0, 4):
        series_x.append(np.array(range(0, 20)))

    # Fill the y-values of the time-series (actual values)
    series_y.append(np.array(range(3, 23)))
    series_y.append(np.array(range(4, 24)))
    series_y.append(np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 13, 14, 15, 16, 17, 18, 19, 20, 21]))
    series_y.append(np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 11, 12, 13, 14, 15, 16, 17, 18, 19]))

    # Normalize, focus on shape
    series_y = [preprocessing.scale(v, axis=1) for v in series_y]

    # show input timeseries
    label = 0
    for i in series_y:
        plt.plot(i, label='timeries' + str(label))
        label += 1
    plt.legend()
    plt.show()

    # Perform clustering
    result = hierarchical([series_l, series_x, series_y], split_ratio=0.5, depth=1, min_size=1, plot=False)
    for cluster_i, cluster in enumerate(flat_clusters(result)):
        print("Cluster {}: ".format(cluster_i) + " ".join([str(i) for i in cluster]))


if __name__ == "__main__":
    test()
