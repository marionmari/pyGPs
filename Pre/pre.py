#===============================================================================
#    Copyright (C) 2013
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [marthaler at ge dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at iais dot fraunhofer dot de]
# 
#    Fraunhofer IAIS, STREAM Project, Sankt Augustin, Germany
# 
#    This file is part of pyGPs.
# 
#    pyGPs is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
# 
#    pyGPs is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, see <http://www.gnu.org/licenses/>.
#===============================================================================

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