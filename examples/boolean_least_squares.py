#!/usr/bin/python
import numpy as np
import cvxpy as cvx
from qcqp import *

n, m = 10, 15
np.random.seed(1)

A = np.random.randn(m, n)
b = np.random.randn(m, 1)

x = cvx.Variable(n)
obj = cvx.sum_squares(A*x - b)
cons = [cvx.square(x) == 1]
prob = cvx.Problem(cvx.Minimize(obj), cons)
qcqp = QCQP(prob)

# sample from the semidefinite relaxation
qcqp.suggest(SDR)
print("SDR lower bound: %.3f" % qcqp.sdr_bound)

f_cd, v_cd = qcqp.improve(COORD_DESCENT)
x_cd = x.value
print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))

# SDR solution is cached and not solved again
qcqp.suggest(SDR)
f_dccp, v_dccp = qcqp.improve(DCCP)
print("Penalty CCP: objective %.3f, violation %.3f" % (f_dccp, v_dccp))
f_dccp, v_dccp = qcqp.improve(COORD_DESCENT, phase1=False)
x_dccp = x.value
print("Penalty CCP + coordinate descent: objective %.3f, violation %.3f" % (f_dccp, v_dccp))

qcqp.suggest(SDR)
f_admm, v_admm = qcqp.improve(COORD_DESCENT)
f_admm, v_admm = qcqp.improve(ADMM, phase1=False)
x_admm = x.value
print("Coordinate descent + nonconvex ADMM: objective %.3f, violation %.3f" % (f_admm, v_admm))
