Semi-supervised Learning with Graph
=======================================

The code shown in this tutorial can be executed by running *pyGPs/Demo/demo_KernelOnGraph.py*


Load data
--------------------
We used the same dataset from GPMC example. i.e. The USPS digits dataset [1]_.
Each digit of :math:`16*16` pixels is flattened into a :math:`256` dimension vector.
For the simplicity of demo, we only selected digits :math:`1` s and :math:`7` s such that we have a binary classification problem where digit :math:`1` for class +1 and digit :math:`7` for class -1. Binary classification of digits is a easy task with regard to this USPS digit dataset. Therefore, to increase level of difficulty, we choosed digit :math:`1` s and :math:`7` s instead of other combinations because 1 and 7 are most similar in our dataset. We also reduced the dataset into :math:`100` samples per digit, where the original dataset consist of thousands of samples for each digit.

Here are samples for two digits for :math:`1`

.. image:: _images/digit1_1.png
   :width: 30% 

.. image:: _images/digit1_2.png
   :width: 30% 

and samples for two digits for :math:`7`

.. image:: _images/digit7_1.png
   :width: 30% 

.. image:: _images/digit7_2.png
   :width: 30% 


Form a nearest neighbour graph
--------------------------------
We form a nearest-neighbor graph based on Euclidean distance of the vector representation of digits. Neighboring images have small Euclidean distance. Each digit is a node in the graph. There is an edge if digit :math:`i` is the k-nearest neighbour of digit :math:`j`. We form a symmetrized graph such that we connect nodes :math:`j`, :math:`i` if i is in j’s kNN and vice versa, and therefore a node can have more than k edges. You should import the corresponding module from *pyGPs.GraphStuff* ::

    x,y = load_binary(1,7,reduce=True)
    A = form_knn_graph(x,2)

A is the adjacency matrix of this :math:`2-NN` graph.

Below shows an example of such symmetrized Euclidean :math:`2-NN` graph on some 1s and 2s taking from Xiaojin Zhu's doctoral thesis [2]_.

.. figure:: _images/2nnGraph.png
   :align: center


Kernel on graph
------------------
Several classical kernels on graph described in `Structured Kernels`_ can be built from adjacency matrix :math:`A`. We use diffusion kernel for this example to get the precomputed kernel matrix. ::

    Matrix = diffKernel(A)

.. _Structured Kernels: Graph.html

This a big square matrix with all rows and columns of the number of data points.
By specifying the indice of training data and test data, we will form two matrix M1 and M2 with the exact format which *pyGPs.Core.cov.Pre* needed. ::

    M1,M2 = form_kernel_matrix(Matrix, indice_train, indice_test)

M1 is a matrix with shape **number of training points plus 1** by **number of test points** 
 - cross covariances matrix (train by test) 
 - last row is self covariances (diagonal of test by test)
M2 is a square matrix with **number of training points** for each dimension
 - training set covariance matrix (train by train)  


GP classification
-----------------------
Every ingredients for a basic semi-supervised learning is prepared now.  Lets see how to proceed for :math:`GP` classification. First, the normal way with rbf kernel we have seen several times ::

        model = gp.GPC()
        k = cov.RBF()
        model.setPrior(kernel=k)

Then lets use our kernel precomputed matrix. If you only use precomputed kernel matrix, there is no training data.
However you still need to specify :math:`x` just to fit in the usage of pyGPs for generality reason. 
You can create any :math:`x` as long as the dimension is correct. ::

        x = np.zeros((n,1))
        k = cov.Pre(M1,M2) + cov.RBF()
        model.setPrior(kernel=k)

Moreover, you can composite a kernel for both precomputed matrix and regular kernel function if necessary. ::

        k = cov.Pre(M1,M2) + cov.RBF()
        model.setPrior(kernel=k)

The rest is exactly the same as the demo of GP classification.


Result
-----------------------
For our manually created graph data, an rbf kernel works better than a diffusion kernel on the graph (higher accuracy). The performance in general should depend on the application as well as features of data.

The left image shows the digit that using diffusion kernel will predict the wrong result (should be :math:`7`), 
but rbf kernel does the job fine. The right image shows the digit that all the choices of kernels stated previously will lead to a wrong predictioin (should be :math:`1`).

.. image:: _images/digitBadForDiffu.png
   :width: 30% 

.. image:: _images/digitBadBoth.png
   :width: 30% 


.. [1] A Database for Handwritten Text Recognition Research, J. J. Hull, IEEE PAMI 16(5) 550-554, 1994.
.. [2] Semi-Supervised Learning with Graphs, Xiaojin Zhu, CMU-LTI-05-192, 2005
