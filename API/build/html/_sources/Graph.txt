Kernels on Graphs
============================


Demo
---------------------------------------

.. toctree::
   :maxdepth: 1
   
   SemiSupervised

Functionality
---------------------------------------
Implemented kernels on graphs: 

======================  =================================
diffKernel 	  	Diffusion kernel	  	  	 
VNDKernel		Von-Neumann diffusion kernel
psInvLapKernel  	Pseudo-Inverse of the Laplacian
regLapKernel    	Regularized Laplacian kernel
rwKernel      	        p-step random walk kernel	
cosKernel               Inverse Cosine Kernel
======================  =================================



Graph Kernels
============================

Demo
---------------------------------------

.. toctree::
   :maxdepth: 1
   
   GraphKernel

Functionality
---------------------------------------
Implemented graph kernels:

==================  =====================================
propagationKernel   Propagation kernel
==================  =====================================


Graph Utilities
============================

Functionality
---------------------------------------
Implemented graph utilities:

==================  =======================================================
formKnnGraph   	    create knn graph from vector-valued data
formKernelMatrix    transfer precomputed kernel matrix to pyGPs format
normalizeKernel     normalize kernel matrix
==================  =======================================================
