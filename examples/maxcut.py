#!/usr/bin/python
import numpy as np
import cvxpy as cvx
from qcqp import *

n = 25
np.random.seed(1)

# Make adjacency matrix.
p = 0.2
W = np.asmatrix(np.random.uniform(low=0.0, high=1.0, size=(n, n)))
for i in range(n):
    W[i, i] = 1
    for j in range(i+1, n):
        W[j, i] = W[i, j]
W = (W < p).astype(float)

x = cvx.Variable(n)
obj = 0.25*(cvx.sum_entries(W) - cvx.quad_form(x, W))
cons = [cvx.square(x) == 1]
prob = cvx.Problem(cvx.Maximize(obj), cons)
qcqp = QCQP(prob)

# sample from the semidefinite relaxation
qcqp.suggest(SDR)
print("SDR-based upper bound: %.3f" % qcqp.sdr_bound)

f_cd, v_cd = qcqp.improve(COORD_DESCENT)
print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))

# SDR solution is cached and not solved again
qcqp.suggest(SDR)
f_dccp, v_dccp = qcqp.improve(DCCP, tau=1)
print("Penalty CCP: objective %.3f, violation %.3f" % (f_dccp, v_dccp))

qcqp.suggest(SDR)
f_admm, v_admm = qcqp.improve(ADMM)
print("Nonconvex ADMM: objective %.3f, violation %.3f" % (f_admm, v_admm))
