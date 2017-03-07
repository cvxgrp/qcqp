#!/usr/bin/python
import numpy as np
import cvxpy as cvx
import qcqp

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

# Objective function
f = lambda x: 0.25*(np.sum(W) - (x_round.T*W*x_round)[0, 0])
def violation(x):
    return np.max(np.abs(np.square(x) - 1))

# SDP-based upper bound
ub = prob.solve(method='sdp-relax', solver=cvx.MOSEK)
print ('Upper bound: %.3f' % ub)

# Lower bounds
print ('(objective, maximum violation):')
f_dccp = prob.solve(method='qcqp-dccp', use_sdp=False, solver=cvx.MOSEK, num_samples=10, tau=1)
print ('  Convex-concave programming: (%.3f, %.3f)' % (f_dccp, violation(x.value)))
x_round = np.sign(np.random.randn(n, 1))
f_simple = f(x_round)
print ('  Random feasible point: (%.3f, %.3f)' % (f_simple, violation(x_round)))
f_cd = prob.solve(method='coord-descent', use_sdp=False, num_samples=10)
print ('  Coordinate descent: (%.3f, %.3f)' % (f_cd, violation(x.value)))
f_admm = prob.solve(method='qcqp-admm', use_sdp=False, num_samples=10, num_iters=100)
print ('  Nonconvex ADMM: (%.3f, %.3f)' % (f_admm, violation(x.value)))
