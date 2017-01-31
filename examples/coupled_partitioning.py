#!/usr/bin/python
import numpy as np
from cvxpy import *
import qcqp

n = 3
np.random.seed(1)

V = np.asmatrix(np.random.randn(n, n))
W = np.asmatrix(np.random.randn(n, n))

x = Variable(n)
y = Variable(n)
obj = Minimize(quad_form(x, V) + quad_form(x, W))
cons = [
    square(x) == x,
    square(y) == y,
    0.2*n <=     x.T*y,         x.T*y     <= 0.3*n,
    0.2*n <= (1-x).T*y,     (1-x).T*y     <= 0.3*n,
    0.2*n <=     x.T*(1-y),     x.T*(1-y) <= 0.3*n,
    0.2*n <= (1-x).T*(1-y), (1-x).T*(1-y) <= 0.3*n
]
prob = Problem(obj, cons)

# SDP-based lower bound
lb = prob.solve(method='sdp-relax', solver=MOSEK)

# Upper bounds
ub_cd = prob.solve(method='coord-descent', solver=MOSEK, num_samples=10)
ub_admm = prob.solve(method='qcqp-admm', solver=MOSEK, num_samples=10)
ub_dccp = prob.solve(method='qcqp-dccp', solver=MOSEK, num_samples=10)
print ('Lower bound: %.3f' % lb)
print ('Upper bounds:')
print ('  Coordinate descent: %.3f' % ub_cd)
print ('  Nonconvex ADMM: %.3f' % ub_admm)
print ('  Convex-concave programming: %.3f' % ub_dccp)
