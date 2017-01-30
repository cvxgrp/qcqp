#!/usr/bin/python
import numpy as np
import cvxpy as cvx
import qcqp

def get_cost_matrix():
    p = 0.2
    W = np.asmatrix(np.random.uniform(low=0.0, high=1.0, size=(n, n)))
    for i in range(n):
        W[i, i] = 1
        for j in range(i+1, n):
            W[j, i] = W[i, j]
    W = (W < p).astype(float)
    return W

n = 5
np.random.seed(1)

V = get_cost_matrix()
W = get_cost_matrix()

x = cvx.Variable(n)
y = cvx.Variable(n)
obj = cvx.Minimize(cvx.quad_form(x, V) + cvx.quad_form(x, W))
cons = [
    cvx.square(x) == x,
    cvx.square(y) == y,
    0.2*n <=     x.T*y,         x.T*y     <= 0.3*n,
    0.2*n <= (1-x).T*y,     (1-x).T*y     <= 0.3*n,
    0.2*n <=     x.T*(1-y),     x.T*(1-y) <= 0.3*n,
    0.2*n <= (1-x).T*(1-y), (1-x).T*(1-y) <= 0.3*n
]
prob = cvx.Problem(obj, cons)

# SDP-based lower bound
lb = prob.solve(method='sdp-relax', solver=cvx.MOSEK)

# Upper bounds
ub_cd = prob.solve(method='coord-descent', solver=cvx.MOSEK, num_samples=10)
ub_admm = prob.solve(method='qcqp-admm', solver=cvx.MOSEK, num_samples=10)
ub_dccp = prob.solve(method='qcqp-dccp', solver=cvx.MOSEK, num_samples=10, tau=1)
print ('Lower bound: %.3f' % lb)
print ('Upper bounds:')
print ('  Coordinate descent: %.3f' % lb_cd)
print ('  Nonconvex ADMM: %.3f' % lb_admm)
print ('  Convex-concave programming: %.3f' % lb_dccp)
