#!/usr/bin/python
import numpy as np
import cvxpy as cvx
import qcqp

n, m = 100, 150
np.random.seed(1)

A = np.random.randn(m, n)
b = np.random.randn(m, 1)

x = cvx.Variable(n)
obj = cvx.sum_squares(A*x - b)
cons = [cvx.square(x) == 1]
prob = cvx.Problem(cvx.Minimize(obj), cons)

# Objective function
f = lambda x: np.sum(np.square(A*x - b))
def violation(x):
    return np.max(np.abs(np.square(x) - 1))

# SDP-based lower bound
lb = prob.solve(method='sdp-relax', solver=cvx.MOSEK)
print ('Upper bound: %.3f' % lb)

# Upper bounds
print ('(objective, maximum violation):')
f_dccp = prob.solve(method='qcqp-dccp', use_sdp=False, solver=cvx.MOSEK, num_samples=100, tau=1)
print ('  Convex-concave programming: (%.3f, %.3f)' % (f_dccp, violation(x.value)))
x_round = np.sign(np.random.randn(n, 1))
f_simple = f(x_round)
print ('  Random feasible point: (%.3f, %.3f)' % (f_simple, violation(x_round)))
f_cd = prob.solve(method='coord-descent', use_sdp=False, num_samples=100)
print ('  Coordinate descent: (%.3f, %.3f)' % (f_cd, violation(x.value)))
f_admm = prob.solve(method='qcqp-admm', use_sdp=False, num_samples=100, num_iters=100)
print ('  Nonconvex ADMM: (%.3f, %.3f)' % (f_admm, violation(x.value)))
