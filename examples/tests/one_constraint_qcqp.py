#!/usr/bin/python
import numpy as np
import cvxpy as cvx
import scipy.sparse as sp
import qcqp
from qcqp import utilities as u

n = 10
z = np.asmatrix(np.random.randn(n, 1))
A = np.asmatrix(np.random.randn(n, n))
A = (A+A.T)/2
b = np.asmatrix(np.random.randn(n, 1))
c = np.random.randn()

# Solve using SDP relaxation
x = cvx.Variable(n)
obj = cvx.sum_squares(x - z)
cons = [cvx.quad_form(x, A) + b.T*x + c <= 0]
prob = cvx.Problem(cvx.Minimize(obj), cons)
val1 = prob.solve(method='sdp-relax', solver=cvx.MOSEK)

f = u.QuadraticFunction(sp.csr_matrix(A), sp.csc_matrix(b), c, '<=')
x = u.onecons_qcqp(z, f)
val2 = ((x-z).T*(x-z))[0, 0]

assert abs(val1-val2) < 1e-5