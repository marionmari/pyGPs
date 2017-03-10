================================================================================
    Marion Neumann [m dot neumann at wustl dot edu]
    Daniel Marthaler [dan dot marthaler at gmail dot com]
    Shan Huang [schan dot huang at gmail dot com]
    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]

    This file is part of pyGPs.
    The software package is released under the BSD 2-Clause (FreeBSD) License.

    Copyright (c) by
    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 18/02/2014
================================================================================

pyGPs is a Python library for Gaussian Process (GP) Regression and Classification.
Here is an online [documentation](http://www-ai.cs.uni-dortmund.de/weblab/static/api_docs/pyGPs/), where you will find a comprehensive introduction to functionalities and demonstrations. You can also find the same doc locally in `/doc/build/html/index.html`. 

Generally speaking, pyGPs is an object-oriented GPs implementation. The functionality follows roughly the gpml matlab implementation by Carl Edward Rasmussen and Hannes Nickisch (Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2013-01-21). Standard GP regression and classification as well as FITC (sparse GPs) inference is implemented.

Further, pyGPs includes implementations of
- minimize.py implemented in python by Roland Memisevic 2008, following minimize.m which is copyright (C) 1999 - 2006, Carl Edward Rasmussen
- scg.py (Copyright (c) Ian T Nabney (1996-2001))
- brentmin.py (Copyright (c) by Hannes Nickisch 2010-01-10.)

Finally, pyGPs is constantly maintained. If you feel you have some relevant skills and are interested in contributing then please do get in touch. We appreciate any feedback

Installing pyGPs
------------------
You can install via pip (**Recommended!**):
        
    pip install pyGPs 

Alternatively, download the archive and extract it to any local directory. 
Install the package using setup.py:

    python setup.py install

or add the local directory to your PYTHONPATH:

    export PYTHONPATH=$PYTHONPATH:/path/to/local/directory/../parent_folder_of_pyGPs

Requirements
--------------
- python 2.6 or 2.7 or *NEW:* python 3
- scipy (v0.13.0 or later), numpy, and matplotlib: open-source packages for scientific computing using the Python programming language. 


Acknowledgements
--------------
The following persons helped to improve this software: Roman Garnett, Maciej Kurek, Hannes Nickisch, Zhao Xu, and Alejandro Molina.

This work is partly supported by the Fraunhofer ATTRACT fellowship STREAM.

Citation
-------------
To cite pyGps, please use the following BibTex:
```
@article{JMLR:v16:neumann15a,
  author  = {Marion Neumann and Shan Huang and Daniel E. Marthaler and Kristian Kersting},
  title   = {pyGPs -- A Python Library for Gaussian Process Regression and Classification},
  journal = {Journal of Machine Learning Research},
  year    = {2015},
  volume  = {16},
  pages   = {2611-2616},
  url     = {http://jmlr.org/papers/v16/neumann15a.html}
}
```

