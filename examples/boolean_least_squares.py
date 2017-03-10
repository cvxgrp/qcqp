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

# sample from the SDP solution
qcqp.suggest(sdp=True, solver=cvx.MOSEK)
print("SDP-based lower bound: %.3f" % qcqp.sdp_bound)

f_cd, v_cd = qcqp.improve(COORD_DESCENT)
x_cd = x.value
print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))

# SDP solution is cached and not solved again
qcqp.suggest(sdp=True)
f_dccp, v_dccp = qcqp.improve(DCCP)
print("Penalty CCP: objective %.3f, violation %.3f" % (f_dccp, v_dccp))
f_dccp, v_dccp = qcqp.improve(COORD_DESCENT, phase1=False)
x_dccp = x.value
print("Penalty CCP: objective %.3f, violation %.3f" % (f_dccp, v_dccp))

qcqp.suggest(sdp=True)
f_admm, v_admm = qcqp.improve(COORD_DESCENT)
f_admm, v_admm = qcqp.improve(ADMM, phase1=False)
x_admm = x.value
print("Coordinate descent + nonconvex ADMM: objective %.3f, violation %.3f" % (f_admm, v_admm))
