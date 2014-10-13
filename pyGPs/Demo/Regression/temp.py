import numpy as np

#xtrain = np.atleast_2d(np.genfromtxt('/Users/dmarthal/Desktop/gp/matlab/gp_cholqr/Snelson_1D_data/train_inputs')).T
#ytrain = np.atleast_2d(np.genfromtxt('/Users/dmarthal/Desktop/gp/matlab/gp_cholqr/Snelson_1D_data/train_outputs')).T
#xstar = np.atleast_2d(np.genfromtxt('/Users/dmarthal/Desktop/gp/matlab/gp_cholqr/Snelson_1D_data/test_inputs')).T

#np.savez('Snelson_1D_data', xtrain=xtrain, ytrain=ytrain, xstar=xstar)

demoData = np.load('Snelson_1D_data.npz')
x = demoData['xtrain']       # training data
y = demoData['ytrain']       # training target
z = demoData['xstar']        # test data

print x.shape, y.shape, z.shape
