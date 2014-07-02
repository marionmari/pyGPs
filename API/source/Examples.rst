Demos
=========================
There are several demos exemplifying the use of pyGPs for various Gaussian process (:math:`GP`) tasks. 
We recommend to first go through *Basic GP Regression* which introduces the :math:`GP` regression model. 
Basic regression is the most intuitive and simplest learning task feasable with :math:`GPs`. 
The other demos will then provide a general insight into more advanced functionalities of the package. 
You will also find the implementation of the demos in the source_ folder under `pyGPs/Demo`_.

.. _pyGPs/Demo: https://github.com/marionmari/pyGPs/tree/master/pyGPs/Demo

The Demos give some theoretical explanations. Further, it is useful to have a look at our documentation on `Kernels & Means`_ and `Optimizers`_. 

.. _Kernels & Means: Kernels.html
.. _Optimizers: Opts.html

.. _source: https://github.com/marionmari/pyGPs

Regression

.. toctree::
   :maxdepth: 1

   GPR
   GPR_FITC

Classification

.. toctree::
   :maxdepth: 1

   GPC
   GPC_FITC
   GPMC

Some examples for real-world data

.. toctree::
   :maxdepth: 1
   
   CV
   demoMaunaLoa
   demoHousing
   SemiSupervised


