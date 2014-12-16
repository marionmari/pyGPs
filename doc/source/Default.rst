List of Functions and Default Parameters
===========================================

Mathematical Definitions of Standard Kernels
---------------------------------------------
**Constant kernel**: 
:math:`k(x,x') = \sigma^2`

**Gabor kernel**: 
:math:`k(x,x') = \exp{(-\frac{||x-x'||^2} {2l^2})}  \cos{(\frac{2\pi||x-x'||}{p})}`

**Linear kernel with ARD**: 
:math:`k(x,x') = \sigma^2 x^T L^{-1} x'` 

where :math:`L` is a diagnal matrix 
consist of length scale :math:`l_{i}^2` for each dimension :math:`i`.

**Linear kernel**: 
:math:`k(x,x') = \sigma^2 x^T x'`


**Matern kernel**: 
:math:`k(x,x') = \sigma^2 f(r\sqrt{d})\exp{(-r\sqrt{d})}` 

with 
:math:`f(t)=1` for :math:`d=1`,
:math:`f(t)=1+t` for :math:`d=3` and 
:math:`f(t)=(1+t+t^2)/3` for :math:`d=5`.

where r is the distance 
:math:`r=\sqrt{||x-x'||^T L^{-1}||x-x'||}` and :math:`L` is a diagnal matrix 
consist of length scale :math:`l_{i}^2` for each dimension :math:`i`. 

**Independent noise kernel**: 
:math:`k(x,x') = \sigma_{n}^{2}`

**Periodic kernel**: 
:math:`k(x,x') = \sigma^2\exp{(-\frac{2\sin^2(\pi||x-x'||/p)}{l^2})}`

**Piecewise polynomial kernel**: 
:math:`k(x,x') = \sigma^2 \max{(1-r,0)}^{(j+v)} f(r,j)`

with :math:`j = floor(D/2)+v+1`

where D is the dimension of input and 
:math:`L` is a diagnal matrix 
consist of length scale :math:`l_{i}^2` for each dimension :math:`i` 
and f is a function depending on v. See gpml matlab v3.4.

**Polynomial kernel**: 
:math:`k(x,x') = \sigma^2 (x^T x' + c )^d`

**Squared exponential kernel**: 
:math:`k(x,x') = \sigma^2 \exp{(-\frac{||x-x'||^2}{2l^2})}`

**Squared exponential kernel with ARD**: 
:math:`k(x,x') = \sigma^2 \exp{(-\frac{||x-x'||^T L^{-1}||x-x'||}{2})}`

where :math:`L` is a diagnal matrix 
consist of length scale :math:`l_{i}^2` for each dimension :math:`i`.

**Squared exponential kernel with unit magnitude**: 
:math:`k(x,x') = \exp{(-\frac{||x-x'||^2}{2l^2})}`

**Rational quadratic kernel**: 
:math:`k(x,x') = \sigma^2 (1+\frac{||x-x'||^2}{2\alpha l^2})^{-\alpha}`

**Rational quadratic kernel with ARD**: 
:math:`k(x,x') = \sigma^2 (1+\frac{||x-x'||^T L^{-1}||x-x'||}{2\alpha})^{-\alpha}`

where :math:`L` is a diagnal matrix 
consist of length scale :math:`l_{i}^2` for each dimension :math:`i`.



List of Kernels and Default Parameters
---------------------------------------

.. automodule:: pyGPs.Core.cov
   :members:


List of Means and Default Parameters
-----------------------------------------
 
.. automodule:: pyGPs.Core.mean
   :members:


List of Likelihoods 
---------------------------------------

.. autoclass:: pyGPs.Core.lik.Erf
   :members:

.. autoclass:: pyGPs.Core.lik.Gauss
   :members:

.. autoclass:: pyGPs.Core.lik.Laplace
   :members:

List of Inference 
-----------------------------------------

.. autoclass:: pyGPs.Core.inf.Exact
   :members:

.. autoclass:: pyGPs.Core.inf.EP
   :members:

.. autoclass:: pyGPs.Core.inf.Laplace
   :members:

.. autoclass:: pyGPs.Core.inf.FITC_Exact
   :members:

.. autoclass:: pyGPs.Core.inf.FITC_EP
   :members:

.. autoclass:: pyGPs.Core.inf.FITC_Laplace
   :members:

