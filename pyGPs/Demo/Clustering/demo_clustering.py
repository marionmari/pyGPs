__author__ = 'christiaanleysen'

# This example divides a set of timeseries into two clusters of the most similar timeseries using the general 
# model learn over a set of timeseries.
#
# Find more information in the foloowing paper: 
#
# "Energy consumption profiling using Gaussian Processes",
# Christiaan Leysen*, Mathias Verbeke†, Pierre Dagnely†, Wannes Meert* 
# *Dept. Computer Science, KU Leuven, Belgium
# †Data Innovation Team, Sirris, Belgium
# https://lirias.kuleuven.be/bitstream/123456789/550688/1/conf2.pdf
import pyGPs,math
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from pyGPs.Demo.Clustering import pyGP_extension as GPE
import numpy as np
import matplotlib.pyplot as plt
import math


def calculateRMSEPyGP(vectorX, vectorY, labelList, weighted=True):
    """
    calculate the root mean squared error
    Parameters:
    -----------
    vectorX: timestamps of the timeseries
    vectorY: valueSet of the timeseries
    labelList: labels of the timeseries
    weighted: weight RMSE wrt variance of prediction
    Returns:
    --------
    list of (household,rmse) tuples
    """
    #setX = [preprocessing.scale(element )for element in vectorX]
    setY = preprocessing.scale(vectorY,axis=1)

    model = pyGPs.GPR()      # specify model (GP regression)
    k =  pyGPs.cov.Linear() + pyGPs.cov.RBF() #hyperparams will be set with optimizeHyperparameters method
    model.setPrior(kernel=k)

    hyperparams, model2 = GPE.optimizeHyperparameters([0.0000001,0.0000001,0.0000001],model,vectorX,setY,bounds=[(None,5),(None,5),(None,5)],method = 'L-BFGS-B')
    print('hyerparameters used:',hyperparams)
    # mean (y_pred) variance (ys2), latent mean (fmu) variance (fs2), log predictive prob (lp)
    y_pred, ys2, fm, fs2, lp = model2.predict(vectorX[0])

    #plot general model after normalizing the input timeseries
    xs = vectorX[0]
    ym = y_pred
    xss  = np.reshape(xs,(xs.shape[0],))
    ymm  = np.reshape(ym,(ym.shape[0],))
    ys22 = np.reshape(ys2,(ys2.shape[0],))
    plt.plot(xss, ym, color='red', label="Prediction")
    plt.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=[0.7539, 0.89453125, 0.62890625, 1.0], linewidths=0.0)
    for i in setY:
        plt.plot(i,color='blue')
    plt.legend()
    plt.show(block=True)

    rmseData = []
    for i in range(0,len(vectorY),1):
        if weighted:
            rmse = math.sqrt(mean_squared_error(vectorY[i], y_pred, 1.1*np.max(ys2)-ys2))
        else:
            rmse = math.sqrt(mean_squared_error(vectorY[i], y_pred))
        HH = labelList[i]
        rmseData.append((HH,rmse))
    return rmseData


def divideInClusters(clusterlist,labelList,threshold,clusterSize,splitRatio):
    """
    aux method for the clustering which divides the clusterlist further into clusters using a certain threshold
    Parameters:
    -----------

    clusterlist: list with timeseries which needs to be clustered
    labelList: list with the labels of the timeseries
    threshold: mean similarity threshold to divide the clusters
    clusterSize: minimum cluster size
    splitratio: ratio of timeseries that will be devided into a new cluster
    Returns:
    --------
    list of clusters of (household,rmse) tuples
    """
    vectorX,vectorY = clusterlist[0],clusterlist[1]

    listRMSE = calculateRMSEPyGP(vectorX, vectorY, labelList, weighted=True)
    sortedListRMSE = sorted(listRMSE, key=lambda x: x[1])

    NormalizeValue = sortedListRMSE[-1][1]
    sortedListRMSE_normalized = [(x[0],x[1] / NormalizeValue) for x in sortedListRMSE][::-1]

    clusterSizeLength = int(math.ceil(splitRatio * len(sortedListRMSE_normalized)))

    newClusterlist = sortedListRMSE_normalized[-clusterSizeLength:]#[::-1]
    newRemaininglist = sortedListRMSE_normalized[:len(sortedListRMSE_normalized)-clusterSizeLength]#[::-1]

    if (len(newClusterlist)>=clusterSize and len(newRemaininglist)>=clusterSize):
        if(np.mean([item[1] for item in sortedListRMSE_normalized])<threshold): #check goodness of cluster
            printClusterList = [item[0] for item in newClusterlist]
            printRemainingList = [item[0] for item in newRemaininglist]
            return newClusterlist,newRemaininglist,printClusterList,printRemainingList
        else:
            printClusterList = [item for item in labelList][::-1]
            return clusterlist, [],printClusterList,[]
    else:
        printClusterList = [item[0] for item in clusterlist]
        return clusterlist, [],printClusterList,[]


def test():
    vectorX =[]
    vectorY =[]
    labelList = []
    #Fill the x-values of the timeseries with a time between 0 and 20
    for i in range(0,4):
        vectorX.append(np.array(range(0,20)))
        labelList.append("timeseries"+str(i))
    print(vectorX)

    #Fill the y-values of the timeseries (actual values)
    vectorY.append(np.array(range(3,23)))
    vectorY.append(np.array(range(4,24)))
    vectorY.append(np.array([2,2,2,2,2,2,2,2,2,2,2,13,14,15,16,17,18,19,20,21]))
    vectorY.append(np.array([3,3,3,3,3,3,3,3,3,3,3,11,12,13,14,15,16,17,18,19]))


    #show input timeseries
    label = 0
    for i in vectorY:
        plt.plot(i,label='timeries'+str(label))
        label += 1
    plt.legend()
    plt.show()

    #choose cluster parameters
    splitRatio = 0.5 #splitsings ratio of the clusters
    minClusterSize=1 #Minimum cluster size
    meanSimilarityThreshold = 0.9 #similarity threshold

    newClusterlist,newRemaininglist,printClusterList,printRemainingList = divideInClusters([vectorX,vectorY],labelList,meanSimilarityThreshold,minClusterSize,splitRatio)

    print("cluster1: "+str(np.sort(printClusterList)))
    print("cluster2: "+str(np.sort(printRemainingList)))


if __name__ == "__main__":
    test()

