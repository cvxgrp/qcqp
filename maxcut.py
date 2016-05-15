#!/usr/bin/python
import cvxpy as cvx
import qcqp
import numpy as np

n = 20
np.random.seed(1)

# Make adjacency matrix.
W = np.random.binomial(1, 0.2, size=(n, n))
W = np.asmatrix(W)

x = cvx.Variable(n)
v = np.asmatrix(np.random.randn(1, n))
obj = 0.25*(cvx.sum_entries(W) - cvx.quad_form(x, W))
cons = [cvx.square(x) == 1]
prob = cvx.Problem(cvx.Maximize(obj), cons)

ub = prob.solve(method='relax-SDP', solver=cvx.MOSEK)
print ('Upper bound: %.3f' % ub)

x_round = np.sign(np.random.randn(n, 1))
lb = 0.25*(np.sum(W) - (x_round.T*W*x_round)[0, 0])
print ('Lower bound: %.0f' % lb)


#print (W)
#print (x.value)
#print (x_round)
