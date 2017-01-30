#!/usr/bin/python

# Secondary user multicast beamforming
#   minimize ||w||^2
#   subject to |h_i^H w|^2 >= tau
#              |g_i^H w|^2 <= eta
# with variable w in complex^n

import numpy as np
import cvxpy as cvx
import qcqp

n = 10
m = 8
l = 2

tau = 10
eta = 1

np.random.seed(1)
H = np.random.randn(m, n)
G = np.random.randn(l, n)

w = cvx.Variable(n)
obj = cvx.Minimize(cvx.sum_squares(w))
cons = [cvx.square(H*w) >= tau, cvx.square(G*w) <= eta]
prob = cvx.Problem(obj, cons)

# SDP-based lower bound
lb = prob.solve(method='sdp-relax', solver=cvx.MOSEK)
print ('Lower bound: %.3f' % lb)

# Upper bounds
ub_cd = prob.solve(method='coord-descent', solver=cvx.MOSEK, num_samples=10)
ub_admm = prob.solve(method='qcqp-admm', solver=cvx.MOSEK, num_samples=10)
ub_dccp = prob.solve(method='qcqp-dccp', solver=cvx.MOSEK, num_samples=10, tau=1)
print ('Lower bound: %.3f' % lb)
print ('Upper bounds:')
print ('  Coordinate descent: %.3f' % ub_cd)
print ('  Nonconvex ADMM: %.3f' % ub_admm)
print ('  Convex-concave programming: %.3f' % ub_dccp)
