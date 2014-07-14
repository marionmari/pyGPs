================================================================================
    Marion Neumann [marion dot neumann at uni-bonn dot de]
    Daniel Marthaler [marthaler at ge dot com]
    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]

    This file is part of pyGPs.
    The software package is released under the BSD 2-Clause (FreeBSD) License.

    Copyright (c) by
    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 18/02/2014
================================================================================

pyGPs is a library containing code for Gaussian Process (GP) Regression and Classification.

pyGPs is an object-oriented implementation of GPs. Its functionalities follow roughly the gpml matlab implementaion by Carl Edward Rasmussen and Hannes Nickisch (Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2013-01-21).

Standard GP regression and (binary) classification as well as FITC (spares GPs) inference is implemented.
For a list of implemented covariance, mean, likelihood, and inference functions see list_of_functions.txt. 
The current implementation is optimized and tested, however, the work on this library is still in progress. We appreciate any feedback.

For a comprehensive introduction to functionalities and demonstrations can be found in the *doc* folder; just open /doc/build/html/index.html in your browser to get to the html documentation of the whole package. 

Further, pyGPs includes implementations of
- minimize.py implemented in python by Roland Memisevic 2008, following minimize.m which is copyright (C) 1999 - 2006, Carl Edward Rasmussen
- scg.py (Copyright (c) Ian T Nabney (1996-2001))
- brentmin.py (Copyright (c) by Hannes Nickisch 2010-01-10.)


Installing pyGPs
------------------
Download the archive and extract it to any local directory.

You can either add the local directory to your PYTHONPATH:
    export PYTHONPATH=$PYTHONPATH:/path/to/local/directory/../parent_folder_of_pyGPs

or install the package using setup.py:
    sudo python setup.py install

Requirements
--------------
- python 2.6 or 2.7
- scipy, numpy, and matplotlib: open-source packages for scientific computing using the Python programming language. 


Acknowledgements
--------------
The following persons helped to improve this software: Roman Garnett, Maciej Kurek, Hannes Nickisch, Zhao Xu, and Alejandro Molina.

This work is partly supported by the Fraunhofer ATTRACT fellowship STREAM.

