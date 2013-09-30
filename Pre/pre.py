#================================================================================
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [marthaler at ge dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]
#
#    This file is part of pyGP_OO.
#    The software package is released under the BSD 2-Clause (FreeBSD) License.
#
#    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 30/09/2013
#================================================================================


import numpy as np

def load_data(file):
    '''
    load data file which formated the same as UCL data
    i.e. with one line for each instance,
    attributes are splited by "," and the last attribute is the target
    ''' 
    x = []
    y = []
    with open(file) as f:
        for index,line in enumerate(f):
            feature = line.split(',')
            attr = feature[:-1]
            attr = [float(i) for i in attr]
            target = [feature[-1]]       
            x.append(attr)
            y.append(target)
    x = np.array(x)
    y = np.array(y)
    return x,y