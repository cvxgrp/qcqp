"""
Copyright 2016 Jaehyun Park

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import division
import numpy as np
import scipy.sparse as sp
import cvxpy as cvx

# Encodes a quadratic function x^T P x + q^T x + r,
# with an optional relation operator '<=' or '=='
# so that the function can also encode a constraint.
class QuadraticFunction:
    def __init__(self, P, q, r, relop=None):
        self.P = P
        self.q = q
        self.r = r
        self.relop = relop

    def eval(self, x):
        return (x.T*(self.P*x + self.q) + self.r)[0, 0]

    # Evaluates f with a cvx expression object x.
    def eval_cvx(self, x):
        return cvx.quad_form(x, self.P) + self.q.T*x + self.r

    def violation(self, x):
        assert self.relop is not None
        if self.relop == '==':
            return abs(self.eval(x))
        else:
            return max(0, self.eval(x))

    # Returns the "homogeneous form" matrix M of the function
    # so that (x, 1)^T M (x, 1) is same as f(x).
    def homogeneous_form(self):
        return sp.bmat([[self.P, self.q/2], [self.q.T/2, self.r]])

    # Returns QuadraticFunction f1, f2 such that
    # f(x) = f1(x) - f2(x), with f1 and f2 both convex.
    # Affine and constant components are always put into f1.
    def dc_split(self, use_eigen_split=False):
        n = self.P.shape[0]

        if P.nnz == 0: # P is zero
            P1, P2 = sp.csr_matrix((n, n)), sp.csr_matrix((n, n))
        if use_eigen_split:
            lmb, Q = LA.eigh(self.P.todense())
            P1 = sum([Q[:, i]*lmb[i]*Q[:, i].T for i in range(n) if lmb[i] > 0])
            P2 = sum([-Q[:, i]*lmb[i]*Q[:, i].T for i in range(n) if lmb[i] < 0])
            assert abs(np.sum(P1 - P2 - P)) < 1e-8
        else:
            lmb_min = np.min(LA.eigh(P.todense())[0])
            if lmb_min < 0:
                P1 = P + (1-lmb_min)*sp.identity(n)
                P2 = (1-lmb_min)*sp.identity(n)
            else:
                P1 = P
                P2 = sp.csr_matrix((n, n))
        f1 = QuadraticFunction(P1, self.q, self.r)
        f2 = QuadraticFunction(P2, sp.csc_matrix((n, 1)), 0)
        return (f1, f2)

# given indefinite P
# returns a pair of psd matrices (P+, P-) with P = P+ - P-
def split_quadratic(P, use_eigen_split=False):
    n = P.shape[0]
    # zero matrix
    if P.nnz == 0:
        return (sp.csr_matrix((n, n)), sp.csr_matrix((n, n)))
    if use_eigen_split:
        lmb, Q = LA.eigh(P.todense())
        Pp = sum([Q[:, i]*lmb[i]*Q[:, i].T for i in range(n) if lmb[i] > 0])
        Pm = sum([-Q[:, i]*lmb[i]*Q[:, i].T for i in range(n) if lmb[i] < 0])
        assert abs(np.sum(Pp-Pm-P))<1e-8
        return (Pp, Pm)
    else:
        lmb_min = np.min(LA.eigh(P.todense())[0])
        if lmb_min < 0:
            return (P + (1-lmb_min)*sp.identity(n), (1-lmb_min)*sp.identity(n))
        else:
            return (P, sp.csr_matrix((n, n)))

class QCQP:
    def __init__(self, f0, fs):
        assert all([f.relop is not None for f in fs])
        self.f0 = f0
        self.fs = fs
        self.n = f0.P.shape[0]
        self.m = len(fs)
    def fi(self, i):
        return self.fs[i]
    def violations(self, x): # list of constraint violations
        return [f.violation(x) for f in self.fs]

# given interval I and array of intervals C = [I1, I2, ..., Im]
# returns [I1 cap I, I2 cap I, ..., Im cap I]
def interval_intersection(C, I):
    ret = []
    for J in C:
        IJ = (max(I[0], J[0]), min(I[1], J[1]))
        if IJ[0] <= IJ[1]:
            ret.append(IJ)
    return ret

# TODO: optimize repeated calculations (cache factors, etc.)
def one_qcqp(z, f, tol=1e-6):
    """ Solves a nonconvex problem
      minimize ||x-z||_2^2
      subject to f(x) = x^T P x + q^T x + r ~ 0
      where the relation ~ is given by f.relop (either <= or ==)
    """

    # if constraint is ineq and z is feasible: z is the solution
    if f.relop == '<=' and f.eval(z) <= 0:
        return z

    lmb, Q = map(np.asmatrix, LA.eigh(f.P.todense()))
    zhat = Q.T*z
    qhat = Q.T*f.q

    # now solve a transformed problem
    # minimize ||xhat - zhat||_2^2
    # subject to sum(lmb_i xhat_i^2) + qhat^T xhat + r = 0
    # constraint is now equality from
    # complementary slackness
    def phi(nu):
        xhat = -np.divide(nu*qhat-2*zhat, 2*(1+nu*lmb.T))
        return (lmb*np.power(xhat, 2) + qhat.T*xhat + f.r)[0, 0]
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
    while e-s > tol:
        m = (s+e)/2.
        p = phi(m)
        if p > 0: s = m
        elif p < 0: e = m
        else:
            s = e = m
            break
    nu = (s+e)/2.
    xhat = -np.divide(nu*qhat-2*zhat, 2*(1+nu*lmb.T))
    x = Q*xhat
    return x

# coefs = [(p0, q0, r0), (p1, q1, r1), ..., (pm, qm, rm)]
# returns the optimal point of the following program, or None if infeasible
#   minimize p0 x^2 + q0 x + r0
#   subject to pi x^2 + qi x + ri <= s
# TODO: efficiently find feasible set using BST
def onevar_qcqp(coefs, s, tol=1e-4):
    # feasible set as a collection of disjoint intervals
    C = [(-np.inf, np.inf)]
    for p, q, r in coefs[1:]:
        if p > tol:
            D = q**2 - 4*p*(r-s)
            if D >= 0:
                rD = np.sqrt(D)
                I = ((-q-rD)/(2*p), (-q+rD)/(2*p))
                C = interval_intersection(C, I)
            else: # never feasible
                return None
        elif p < -tol:
            D = q**2 - 4*p*(r-s)
            if D >= 0:
                rD = np.sqrt(D)
                I1 = (-np.inf, (-q-rD)/(2*p))
                I2 = ((-q+rD)/(2*p), np.inf)
                C = interval_intersection(C, I1) + interval_intersection(C, I2)
        else:
            if q > tol:
                I = (-np.inf, (s-r)/q)
            elif q < -tol:
                I = ((s-r)/q, np.inf)
            else:
                continue
            C = interval_intersection(C, I)
    bestx = None
    bestf = np.inf
    (p, q, r) = coefs[0]
    def f(x): return p*x*x + q*x + r
    for I in C:
        # left unbounded
        if I[0] < 0 and np.isinf(I[0]) and (p < 0 or (p < tol and q > 0)):
            return -np.inf
        # right unbounded
        if I[1] > 0 and np.isinf(I[1]) and (p < 0 or (p < tol and q < 0)):
            return np.inf
        (fl, fr) = (f(I[0]), f(I[1]))
        if bestf > fl:
            (bestx, bestf) = I[0], fl
        if bestf > fr:
            (bestx, bestf) = I[1], fr
    # unconstrained minimizer
    if p > tol:
        x0 = -q/(2*p)
        for I in C:
            if I[0] <= x0 and x0 <= I[1]:
                return x0
    return bestx

# given coefficients triples in the form of (p, q, r)
# returns the list of violations of px^2 + qx + r <= 0 constraints
def get_violation_onevar(x, coefs):
    ret = []
    for c in coefs:
        p, q, r = c
        ret.append(max(0, p*x**2 + q*x + r))
    return ret

# regard f(x) as a quadratic expression in xk and returns the coefficients
# where f is an instance of QuadraticFunction
# TODO: speedup
def get_onevar_coeffs(x, k, f):
    z = np.copy(x)
    z[k] = 0
    t2 = f.P[k, k]
    t1 = 2*f.P[k, :]*z + f.q[k, 0]
    t0 = z.T*(f.P*z + f.q) + f.r
    return (t2, t1, t0)

