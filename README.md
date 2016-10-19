QCQP
====

QCQP is a package for modeling and solving quadratically constrained quadratic programs (QCQPs) that are not necessarily convex, using heuristics and relaxations. The methods are discussed in [our associated paper](http://stanford.edu/class/ee364b/lectures/relaxations.pdf).

QCQP is built on top of [CVXPY](http://www.cvxpy.org/), a domain-specific language for convex optimization embedded in Python.

Installation
------------
You should first install CVXPY, following the instructions [here](http://www.cvxpy.org/). If you already have CVXPY, make sure you have the latest version by running ``pip install --upgrade cvxpy``.

The simplest and recommended way of installing QCQP is to run ``pip install qcqp``.
To install the package from source, run ``python setup.py install`` in the source directory.

Example
-------
The following code uses semidefinite programming (SDP) relaxation to get an upper bound on a random instance of the maximum cut problem.
```
n = 20
numpy.random.seed(1)

# Make adjacency matrix.
W = numpy.random.binomial(1, 0.2, size=(n, n))
W = numpy.asmatrix(W)

# Form a nonconvex problem.
x = cvx.Variable(n)
obj = 0.25*(cvx.sum_entries(W) - cvx.quad_form(x, W))
cons = [cvx.square(x) == 1]
prob = cvx.Problem(cvx.Maximize(obj), cons)

# Solve the SDP relaxation.
ub = prob.solve(method='sdp-relax', solver=cvx.MOSEK)
print ('Upper bound: %.3f' % ub)

# Attempt to get a feasible solution.
bestf = prob.solve(method='qcqp-admm')
print (bestf, x.value)
```

Quadratic expressions
---------------------
The quadraticity of an expression ``e`` can be tested using ``e.is_quadratic()``. Below is a list of expressions that CVXPY recognizes as a quadratic expression.
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
In order to use the SDP relaxation heuristic, the problem must have a quadratic objective function and quadratic constraints, using standard CVXPY syntax. Below is a list of available solve methods for QCQPs:
* ``problem.solve(method="sdp-relax")`` solves the SDP relaxation of the problem and returns the SDP lower bound (or an upper bound in the case of maximization problem).
* ``problem.solve(method="qcqp-admm")`` attempts to find a feasible solution via consensus [alternating directions method of multipliers](http://stanford.edu/~boyd/admm.html) (ADMM).
* ``problem.solve(method="qcqp-dccp")`` automatically splits indefinite quadratic functions to convex and concave parts, then invokes the [DCCP](https://github.com/cvxgrp/dccp) package to find a feasible solution.
