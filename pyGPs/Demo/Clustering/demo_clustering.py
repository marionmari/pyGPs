__author__ = 'christiaanleysen'

#This example divides a set of timeseries into two clusters of the most similar timeseries using the general model learn over a set of timeseries



import pyGPs,math
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from Demo.Clustering import pyGP_extension as GPE
import numpy as np
import matplotlib.pyplot as plt



def calculateRMSEPyGP(vectorX,vectorY,labelList):
    """
    calculate the root mean squared error
    Parameters:
    -----------

    vectorX: timestamps of the timeseries
    vectorY: valueSet of the timeseries
    labelList: labels of the timeseries
    Returns:
    --------
    list of (household,rmse) tuples
    """

    #setX = [preprocessing.scale(element )for element in vectorX]
    setY=preprocessing.scale(vectorY,axis=1)

    model = pyGPs.GPR()      # specify model (GP regression)
    k =  pyGPs.cov.Linear() + pyGPs.cov.RBF() #hyperparams will be set with optimizeHyperparameters method
    model.setPrior(kernel=k)



    hyperparams, model2 = GPE.optimizeHyperparameters([0.0000001,0.0000001,0.0000001],model,vectorX,setY,bounds=[(None,5),(None,5),(None,5)],method = 'L-BFGS-B')
    print('hyerparameters used:',hyperparams)

    y_pred, ys2, fm, fs2, lp = model2.predict(vectorX[0])


    #plot general model after normalizing the input timeseries
    plt.plot(y_pred, color='red')
    for i in setY:
        plt.plot(i,color='blue')
    plt.show(block=True)


    rmseData = []
    for i in range(0,len(vectorY),1):
        rmse = mean_squared_error(vectorY[i], y_pred)**0.5
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

    listRMSE = calculateRMSEPyGP(vectorX,vectorY,labelList)


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






