#!/usr/bin/python

# Secondary user multicast beamforming
#   minimize ||w||^2
#   subject to |h_i^H w|^2 >= tau
#              |g_i^H w|^2 <= eta
# with variable w in complex^n.
# Data vectors h_i and g_i are also in complex^n.
# The script below expands out the complex part and
# works with real numbers only.

import numpy as np
import cvxpy as cvx
import qcqp

# n, m, l: 100, 30, 10
n = 10
m = 3
l = 2

tau = 10
eta = 1

#np.random.seed(1)
HR = np.random.randn(m, n)/np.sqrt(2);
HI = np.random.randn(m, n)/np.sqrt(2);
H1 = np.hstack((HR, HI))
H2 = np.hstack((-HI, HR))

GR = np.random.randn(l, n)/np.sqrt(2);
GI = np.random.randn(l, n)/np.sqrt(2);
G1 = np.hstack((GR, GI))
G2 = np.hstack((-GI, GR))

w = cvx.Variable(2*n)
obj = cvx.Minimize(cvx.sum_squares(w))
cons = [
    cvx.square(H1*w) + cvx.square(H2*w) >= tau,
    cvx.square(G1*w) + cvx.square(G2*w) <= eta
]
prob = cvx.Problem(obj, cons)

# SDP-based lower bound
lb = prob.solve(method='sdp-relax', solver=cvx.MOSEK)
print ('Lower bound: %.3f' % lb)

# Upper bounds
print ('Upper bounds:')
ub_admm = prob.solve(method='qcqp-admm', use_sdp=False, solver=cvx.MOSEK, num_samples=10, rho=np.sqrt(m+l), num_iters=1000)
print ('  Nonconvex ADMM: %.3f' % ub_admm)
ub_dccp = prob.solve(method='qcqp-dccp', use_sdp=False, solver=cvx.MOSEK, num_samples=10, tau=1)
print ('  Convex-concave programming: %.3f' % ub_dccp)
ub_cd = prob.solve(method='coord-descent', use_sdp=False, solver=cvx.MOSEK, num_samples=10)
print ('  Coordinate descent: %.3f' % ub_cd)
