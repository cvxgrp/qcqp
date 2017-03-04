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
import qcqp

# n, m, l: 100, 30, 10
n = 50
m = 10
l = 5

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

def violation(x):
    v1 = tau - (np.square(A*x) + np.square(B*x))
    v2 = (np.square(C*x) + np.square(D*x)) - eta
    return max(np.max(v1), np.max(v2), 0)

# SDP-based lower bound
lb = prob.solve(method='sdp-relax', solver=cvx.MOSEK)
print ('Lower bound: %.3f' % lb)

# Upper bounds
print ('(objective, maximum violation):')
f_admm = prob.solve(method='qcqp-admm', use_sdp=False, num_samples=10, rho=np.sqrt(m+l), num_iters=1000, tol=5e-2)
print ('  Nonconvex ADMM: (%.3f, %.3f)' % (f_admm, violation(x.value)))
f_dccp = prob.solve(method='qcqp-dccp', use_sdp=False, num_samples=10)
print ('  Convex-concave programming: (%.3f, %.3f)' % (f_dccp, violation(x.value)))
f_cd = prob.solve(method='coord-descent', use_sdp=False, num_samples=10)
print ('  Coordinate descent: (%.3f, %.3f)' % (f_cd, violation(x.value)))
