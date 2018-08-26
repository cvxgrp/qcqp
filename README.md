QCQP
====

QCQP is a package for modeling and nonconvex solving quadratically constrained quadratic programs (QCQPs) using relaxations and local search heuristics.
Our heuristics are based on the *Suggest-and-Improve* framework:

* *Suggest* method finds a candidate point for a local method.
* *Improve* method takes a point from the *Suggest* method and performs a local search to find a better point.

The notion of better points is defined by the maximum violation of a point and the objective value.
See our [associated paper](https://stanford.edu/~boyd/papers/qcqp.html) for more information on the *Suggest-and-Improve* framework.

QCQP is built on top of [CVXPY](http://www.cvxpy.org/), a domain-specific language for convex optimization embedded in Python.

Installation
------------
*NOTE: QCQP was developed before the release of CVXPY 1.0, which is [not backward compatible](http://www.cvxpy.org/) with the previous version CVXPY 0.4. As of August 2018, QCQP is only compatible with CVXPY 0.4.*

You should first install CVXPY 0.4, following the instructions [here](http://www.cvxpy.org/).
If you already have CVXPY, make sure you have the version compatible with QCQP by running ``conda list cvxpy`` or ``pip show cvxpy``. You can install the compatible version of CVXPY by running the following.
```
conda install -c cvxgrp cvxpy=0.4.9
```

The simplest and recommended way of installing QCQP is to run ``pip install qcqp``.
To install the package from source, run ``python setup.py install`` in the source directory.
You may need to run the commands with the ``sudo`` privilege.

Example
-------
The following code uses semidefinite relaxation (SDR) to get a lower bound on a random instance of the Boolean least squares problem.
Then, using a candidate point generated from the SDR, it runs a coordinate descent method to attempt to find a feasible point with better objective value.
```
from numpy.random import randn
import cvxpy as cvx
from qcqp import *

n, m = 10, 15
A = randn(m, n)
b = randn(m, 1)

# Form a nonconvex problem.
x = cvx.Variable(n)
obj = cvx.sum_squares(A*x - b)
cons = [cvx.square(x) == 1]
prob = cvx.Problem(cvx.Minimize(obj), cons)

# Create a QCQP handler.
qcqp = QCQP(prob)

# Solve the SDP relaxation and get a starting point to a local method
qcqp.suggest(SDR)
print("SDR lower bound: %.3f" % qcqp.sdr_bound)

# Attempt to improve the starting point given by the suggest method
f_cd, v_cd = qcqp.improve(COORD_DESCENT)
print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))
print(x.value)
```

Quadratic expressions
---------------------
The quadraticity of an expression ``e`` can be tested using ``e.is_quadratic()``. Below is a list of expressions that CVXPY recognizes as a quadratic expression. Refer to the [CVXPY documentation](http://www.cvxpy.org/en/latest/tutorial/functions/index.html) for the specifications of the functions.
* Any constant or affine expression
* Any affine transformation applied to a quadratic expression, e.g., ``(quadratic) + (quadratic)`` or ``(constant) * (quadratic)``
* ``(affine) * (affine)``
* ``power(affine, 2)``
* ``square(affine)``
* ``sum_squares(affine)``
* ``quad_over_lin(affine, constant)``
* ``matrix_frac(affine, constant)``
* ``quad_form(affine, constant)``

Constructing and solving problems
---------------------------------
QCQPs must be represented using the standard CVXPY syntax.
In order for the problem to be accepted by ``QCQP``, the problem must have a quadratic objective function and quadratic constraints.
To apply the *Suggest* and *Improve* methods on a QCQP, the corresponding CVXPY problem object must be passed to the QCQP constructor first. For example, if ``problem`` is a CVXPY problem object describing a QCQP, then the following code checks the validity and prepares the *Suggest* and *Improve* methods:
```
qcqp = QCQP(problem)
```

Currently two *Suggest* methods are available for QCQPs:

* ``qcqp.suggest()`` or ``qcqp.suggest(RANDOM)`` fills the values of the variables using independent and identically distributed Gaussian random variables.
* ``qcqp.suggest(SPECTRAL)`` adds all the constraints to a single (possibly nonconvex) constraint, and solves the resulting QCQP with that one constraint. The solution of the relaxation is then stored to the values of the variables. Then, a lower bound (or an upper bound, in the case of a maximization problem) on the optimal value is saved to ``qcqp.spectral_bound``. *The performance of this method is yet to be optimized.*
* ``qcqp.suggest(SDR)`` fills the values of the variables drawn from an optimal probability distribution given by the semidefinite relaxation. Then, a lower bound (or an upper bound, in the case of a maximization problem) on the optimal value is saved to ``qcqp.sdr_bound``. Note: For larger problem instances, ``QCQP`` may fail while solving the semidefinite relaxation. In this case, specifying the MOSEK solver may help: ``qcqp.suggest(SDR, solver=cvx.MOSEK)``. For more information on how to choose solvers, please see the [CVXPY guide](http://www.cvxpy.org/en/latest/tutorial/advanced/index.html#choosing-a-solver).

Below is a list of available solve methods for QCQPs:

* ``qcqp.improve(ADMM)`` attempts to improve the given point via consensus [alternating directions method of multipliers](http://stanford.edu/~boyd/admm.html) (ADMM). An optional parameter ``rho`` can be specified.
* ``qcqp.improve(DCCP)`` automatically splits indefinite quadratic functions to convex and concave parts, then invokes the [DCCP](https://github.com/cvxgrp/dccp) package, using the given point as a starting point. An optional parameter ``tau`` can be specified. In order to use this method, ``DCCP`` must be installed first.
* ``qcqp.improve(COORD_DESCENT)`` performs a two-stage coordinate descent algorithm. The first stage tries to find a feasible point. If a feasible point is found, then the second stage tries to optimize the objective function over the set of feasible points.
* ``qcqp.improve(IPOPT)`` invokes the global optimizer [IPOPT](https://projects.coin-or.org/Ipopt) package, via [PyIpopt](https://github.com/xuy/pyipopt) as the interface. In order to use this method, both ``IPOPT`` and ``PyIpopt`` must be installed first.

Both ``improve()`` and ``suggest()`` methods return a pair ``(f, v)``, where ``f`` represents the current objective value, and ``v`` represents the maximum constraint violation of the current point.

