Installation
============================
1. First, download_ the archive from github and extract it to any local directory.

.. _download: https://github.com/marionmari/pyGPs/archive/1.3.2.tar.gz

2. You can either add the local directory to your PYTHONPATH ::

       export PYTHONPATH=$PYTHONPATH:/path/to/local/directory/../parent_folder_of_pyGPs


3. Or install the package using setup.py::

        sudo python setup.py install

4. Or install via pip::
		
		pip install pyGPs 

Requirements
------------
* `python 2.6 or 2.7`_
* `scipy`_ (v0.13.0 or later), `numpy`_, and `matplotlib`_: open-source packages for scientific computing in Python. 

.. _python 2.6 or 2.7: http://www.python.org/
.. _scipy: http://www.scipy.org/
.. _numpy: http://www.numpy.org/
.. _matplotlib: http://matplotlib.org/


Example installation on Ubuntu & Debian::

	sudo apt-get install python2.7 python-numpy python-scipy python-matplotlib 


Example installation on Mac via Macports (requires XCode and MacPorts)::

	sudo port install python27 py27-numpy py27-scipy py27-matplotlib


For other systems please check the installation instructions on the respective package web sites. 


Testing
---------------
After everything installed, you can check whether they have been installed correctly using pyGPs unit tests by
running the scripts under pyGPs/Testing.
