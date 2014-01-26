from pyGPs.Core import *
import numpy as np

# To have a gerneral idea, 
# you may want to read demo_GPR, demo_kernel and demo_optimization first!
# Here, the focus is on the difference of classification model.

print ''
print '---------------------GPC DEMO-------------------------'

#----------------------------------------------------------------------
# Load demo data (generated from Gaussians)
#----------------------------------------------------------------------
# GPC target class are +1 and -1
demoData = np.load('data_for_demo/classification_data.npz')
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data

# only needed for 2-d contour plotting 
x1 = demoData['x1']          # x for class 1 (with label -1)
x2 = demoData['x2']          # x for class 2 (with label +1)     
t1 = demoData['t1']          # y for class 1 (with label -1)
t2 = demoData['t2']          # y for class 2 (with label +1)
p1 = demoData['p1']          # prior for class 1 (with label -1)
p2 = demoData['p2']          # prior for class 2 (with label +1)



#----------------------------------------------------------------------
# First example -> state default values
#----------------------------------------------------------------------
print 'Basic Example'
model = gp.GPC()             # binary classification (default inference method: EP)
# model.useLikelihood("Logistic")
model.fit(x, y)              # fit default model (mean zero & rbf kernel) with data
model.train(x, y)            # optimize hyperparamters (default optimizer: single run minimize)
model.predict(z)             # predict test cases



#----------------------------------------------------------------------
# GP classification example
#----------------------------------------------------------------------
print 'More Advanced Example'
# Start from a new model 
model = gp.GPC()    

# Analogously to GPR
k = cov.RBFard(log_ell_list=[0.05,0.17], log_sigma=1.)
model.setPrior(kernel=k) 

model.setData(x, y)
model.plotData_2d(x1,x2,t1,t2,p1,p2)

model.fit()
print "Negative log marginal liklihood before:", round(model.nlZ,3)
model.train()
print "Negative log marginal liklihood optimized:", round(model.nlZ,3)

# Prediction
n = z.shape[0]
ymu, ys2, fmu, fs2, lp = model.predict(z, ys=np.ones((n,1)))

# GPC.plot() is a toy method for 2-d data
# plot log probability distribution for class +1
model.plot(x1,x2,t1,t2)


#----------------------------------------------------------------------
# A bit more things you can do
#----------------------------------------------------------------------
# GPC uses Expectation Propagation (EP) inference by default,
# you can explictly change to Laplace Approximation by:
model.useLaplace()

print '--------------------END OF DEMO-----------------------'




