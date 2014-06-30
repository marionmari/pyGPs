#================================================================================
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [marthaler at ge dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]
#
#    This file is part of pyGPs.
#    The software package is released under the BSD 2-Clause (FreeBSD) License.
#
#    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 18/02/2014
#================================================================================

import unittest
import numpy as np
import pyGPs

class ModelTests(unittest.TestCase):

    def setUp(self):
        # random 2d data for testing
        self.x = np.random.normal(loc=0.0, scale=1.0, size=(20,2))





if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()

