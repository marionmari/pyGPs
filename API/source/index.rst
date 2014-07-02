.. pyGPs documentation master file, created by
   sphinx-quickstart on Wed Jul  2 15:51:29 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyGPs's API!
=================================


pyGPs - A Package for Gaussian Processes
===========================================
About the package
---------------------
pyGPs is a library hosting Python implementations of Gaussian processes (GPs) for
machine learning.
pyGPs bridges the gap between systems designed primarily for users, who mainly
want to apply gps and need basic machine learning routines for model training, evaluation, and visualiza-
gp functiontion, and expressive systems for developers, who focus on extending the core
functionalities as covariance and likelihood functions, as well as inference techniques.


The software package is released under the BSD 2-Clause (FreeBSD) License.
:doc:`Copyright <_static/copyright>` (c) by
Marion Neumann, Shan Huang, Daniel Marthaler, & Kristian Kersting, Feb.2014

Further, it includes implementations of

* minimize.py implemented in python by Roland Memisevic 2008, following minimize.m (Copyright (c) Carl Edward Rasmussen (1999-2006))

* scg.py (Copyright (c) Ian T Nabney (1996-2001))

* brentmin.py (Copyright (c) Hannes Nickisch 2010-01-10)

* FITC functionality (following matlab implementations under Copyright (c) by Ed Snelson, Carl Edward Rasmussen and Hannes Nickisch, 2011-11-02)

This is a stable release. If you observe problems or bugs, please let us know.
You can also download a `procedual implementation`_ of GP functionality from Github. However, the procedual version will not be supported in future.

.. _procedual implementation: https://github.com/marionmari/pyGP_PR/  



Authors:
    - Marion Neumann [marion dot neumann at uni-bonn dot de]
    - Shan Huang [shan dot huang at iais dot fraunhofer dot de]
    - Daniel Marthaler [marthaler at ge dot com]
    - Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]

The following persons helped to improve this software: Roman Garnett, Maciej Kurek, Hannes Nickisch, Zhao Xu, and Alejandro Molina.

This work is partly supported by the Fraunhofer ATTRACT fellowship STREAM.


Getting started
------------------------------
.. toctree::
   :maxdepth: 1

   Install 
   Theory 


Tutorials
------------------------------
.. toctree::
   :maxdepth: 1

   Examples

GP functionality
------------------------------
.. toctree::
   :maxdepth: 1

   Kernels
   Likelihoods
   Opts

Extensions
------------------------------
.. toctree::
   :maxdepth: 1

   Graph


API
==================

.. toctree::
   :maxdepth: 1

   modules



