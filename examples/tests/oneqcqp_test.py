from __future__ import division
import warnings
import cvxpy as cvx
import numpy as np
import scipy.sparse as sp
import cvxpy.lin_ops.lin_utils as lu
from numpy import linalg as LA
from cvxpy.utilities import QuadCoeffExtractor

def one_qcqp(z, A, b, c):
    # if z is feasible, it's the solution
    if z.T*A*z + z.T*b + c <= 0:
        return z

    lmb, Q = map(np.asmatrix, LA.eig(A))
    zhat = Q.T*z
    bhat = Q.T*b

    # now solve a transformed problem
    # minimize ||xhat - zhat||_2^2
    # subject to sum(lmb_i xhat_i^2) + bhat^T xhat + c = 0
    # constraint is now equality from
    # complementary slackness
    def phi(nu):
        xhat = -np.divide(nu*bhat-2*zhat, 2*(1+nu*lmb.T))
        return (lmb*np.power(xhat, 2) +
            bhat.T*xhat + c)[0, 0]
    s = -np.inf
    e = np.inf
    for l in np.nditer(lmb):
        if l > 0: s = max(s, -1./l)
        if l < 0: e = min(e, -1./l)
    if s == -np.inf:
        s = -1.
        while phi(s) <= 0: s *= 2.
    if e == np.inf:
        e = 1.
        while phi(e) >= 0: e *= 2.
    eps = 1e-6
    while e-s > eps:
        m = (s+e)/2.
        p = phi(m)
        if p > 0: s = m
        elif p < 0: e = m
        else:
            s = e = m
            break
    nu = (s+e)/2.
    xhat = -np.divide(nu*bhat-2*zhat, 2*(1+nu*lmb.T))
    x = Q*xhat
    return x


import numpy as np
import cvxpy as cvx
import qcqp

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

lb = prob.solve(method='sdp-relax', solver=cvx.MOSEK)
print ('SDP lower bound: %.3f' % lb)

x = one_qcqp(z, A, b, c)
print((x-z).T*(x-z))
