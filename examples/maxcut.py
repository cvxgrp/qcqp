#!/usr/bin/python
import numpy as np
import cvxpy as cvx
import qcqp

n = 15
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

# Objective function
f = lambda x: 0.25*(np.sum(W) - (x_round.T*W*x_round)[0, 0])

# SDP-based upper bound
ub = prob.solve(method='sdp-relax', solver=cvx.MOSEK)

# Random feasible point
x_round = np.sign(np.random.randn(n, 1))

# Lower bounds
lb_simple = f(x_round)
lb_cd = prob.solve(method='coord-descent', solver=cvx.MOSEK, num_samples=10)
lb_admm = prob.solve(method='qcqp-admm', solver=cvx.MOSEK, num_samples=10, num_iters=10, rho=10)
lb_dccp = prob.solve(method='qcqp-dccp', solver=cvx.MOSEK, num_samples=10, tau=1)
print ('Upper bound: %.3f' % ub)
print ('Lower bounds:')
print ('  Random feasible point: %.3f' % lb_simple)
print ('  Coordinate descent: %.3f' % lb_cd)
print ('  Nonconvex ADMM: %.3f' % lb_admm)
print ('  Convex-concave programming: %.3f' % lb_dccp)
