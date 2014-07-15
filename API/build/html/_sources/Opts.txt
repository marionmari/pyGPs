Optimizers
============================

Optimization Methods
----------------------------
As you may have already seen in the demos, the optimizer is initialized in the following way:

.. automethod:: pyGPs.Core.gp.GP.setOptimizer


Developing Optimization Methods
-------------------------------------
We also support the development of new optimizers. 

Your own optimizer should inherent base class Optimizer in *pyGPs.Core.opt*
and follow the template as below: ::

    class MyOptimizer(Optimizer):
        def __init__(self, model=None, searchConfig = None):
            self.model = model

        def findMin(self, x, y):
            '''
            Find minimal value based on negative-log-marginal-likelihood. 
            optimalHyp, funcValue = findMin(x, y)

            where funcValue is the minimal negative-log-marginal-likelihood during optimization,
            and optimalHyp is a flattened numpy array 
            (in sequence of meanfunc.hyp, covfunc.hyp, likfunc.hyp) 
            of the hyparameters to achieve such value.

            You can achieve advanced search strategy by initializing Optimizer with searchConfig, 
            which is an instance of pyGPs.Optimization.conf. 
            See more in pyGPs.Optimization.conf and pyGPs.Core.gp.GP.setOptimizer, 
            as well as in online documentation of section Optimizers.
            '''
            pass



