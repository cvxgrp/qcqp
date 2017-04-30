#!/usr/bin/python

# Secondary user multicast beamforming
#   minimize ||x||^2
#   subject to |h_i^H x|^2 >= tau
#              |g_i^H x|^2 <= eta
# with variable x in complex^n.
# Data vectors h_i and g_i are also in complex^n.
# The script below expands out the complex part and
# works with real numbers only.
# Refer to the companion paper for the details of
# the rewriting.

import numpy as np
import cvxpy as cvx
from qcqp import *

n = 20
m = 5
l = 2

tau = 20
eta = 2

np.random.seed(1)
HR = np.random.randn(m, n)
HI = np.random.randn(m, n)
A = np.hstack((HR, HI))
B = np.hstack((-HI, HR))

GR = np.random.randn(l, n)
GI = np.random.randn(l, n)
C = np.hstack((GR, GI))
D = np.hstack((-GI, GR))

x = cvx.Variable(2*n)
obj = cvx.Minimize(cvx.sum_squares(x))
cons = [
    cvx.square(A*x) + cvx.square(B*x) >= tau,
    cvx.square(C*x) + cvx.square(D*x) <= eta
]
prob = cvx.Problem(obj, cons)
qcqp = QCQP(prob)

# sample from the semidefinite relaxation
qcqp.suggest(SDR)
print("SDR-based lower bound: %.3f" % qcqp.sdr_bound)

# f_cd, v_cd = qcqp.improve(COORD_DESCENT)
# print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))
# f_cd, v_cd = qcqp.improve(ADMM, rho=np.sqrt(m+l), phase1=False)
# print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))

# # SDR solution is cached and not solved again
# qcqp.suggest(SDR)
f_dccp, v_dccp = qcqp.improve(DCCP)
print("Penalty CCP: objective %.3f, violation %.3f" % (f_dccp, v_dccp))

qcqp.suggest(SDR)
f_admm, v_admm = qcqp.improve(COORD_DESCENT)
print("Coordinate descent: objective %.3f, violation %.3f" % (f_admm, v_admm))
f_admm, v_admm = qcqp.improve(ADMM, rho=np.sqrt(m+l))
print("Coordinate descent + ADMM: objective %.3f, violation %.3f" % (f_admm, v_admm))
f_admm, v_admm = qcqp.improve(COORD_DESCENT, phase1=False)
print("Coordinate descent + ADMM + coordinate descent: objective %.3f, violation %.3f" % (f_admm, v_admm))
