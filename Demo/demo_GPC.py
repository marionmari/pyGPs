from pyGP_OO.Core import *
import numpy as np

# We recommend you to read demo_GPR, demo_kernel, and demo_optimization first!
# Here we only focus on the use of different model.

print ''
print '---------------------GPC DEMO-------------------------'

#----------------------------------------------------------------------
# Load demo data (generated from Gaussians)
#----------------------------------------------------------------------
demoData = np.load('data_for_demo/classification_data.npz')
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data
n = z.shape[0]               # number of test points


#----------------------------------------------------------------------
# GP classification example
#----------------------------------------------------------------------