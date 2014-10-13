#================================================================================
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [dan dot marthaler at gmail dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]
#
#    This file is part of pyGPs.
#    The software package is released under the BSD 2-Clause (FreeBSD) License.
#
#    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 18/02/2014
#================================================================================

import pyGPs
import numpy as np

# To have a gerneral idea,
# you may want to read demo_GPR, demo_kernel and demo_optimization first!
# Here, the focus is on the difference of FITC classification.

print ''
print '-------------------GPC_FITC DEMO----------------------'

#----------------------------------------------------------------------
# Load demo data (generated from Gaussians)
#----------------------------------------------------------------------

# GPC_FITC target class are +1 and -1
demoData = np.load('classification_data.npz')
x = np.random.randn(100,1)
y = np.random.randint(0,2,(100,1))
print x.shape, y.shape

#----------------------------------------------------------------------
# Sparse GP classification (FITC) example
#----------------------------------------------------------------------

print '------------------------------------------------------'
print "Example: user-defined inducing points"

model = pyGPs.GPC_FITC()

# You can define inducing points yourself.
# u = np.array([])
u = np.random.randn(10,1)
# and specify inducing point when seting prior
m = pyGPs.mean.Zero()
k = pyGPs.cov.RBFard(log_ell_list=[0.05,0.17], log_sigma=1.)
k = pyGPs.cov.RBF()
model.setPrior(mean=m, kernel=k, inducing_points=u)

# The rest is analogous to what we have done before.
model.setData(x, np.where(y==1,1,-1))
model.fit()
print "Negative log marginal liklihood before optimization:", round(model.nlZ,3)
model.optimize()
print "Negative log marginal liklihood optimized:", round(model.nlZ,3)

# predict
n = u.shape[0]
z = np.atleast_2d(np.linspace(x.min(),x.max(),100)).T
#ymu, ys2, fmu, fs2, lp = model.predict(z, ys=np.ones((n,1)))
ymu, ys2, fmu, fs2, lp = model.predict(z)
plt.plot(ymu)
plt.show()



print '--------------------END OF DEMO-----------------------'




