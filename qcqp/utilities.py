"""
MIT License

Copyright (c) 2017 Jaehyun Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import division
import numpy as np
import scipy.sparse as sp
import cvxpy as cvx
from numpy import linalg as LA
from collections import defaultdict
from itertools import chain
import logging

# Encodes a quadratic function x^T P x + q^T x + r,
# with an optional relation operator '<=' or '=='
# so that the function can also encode a constraint.
#   P is a scipy sparse matrix of size n*n
#   q is a scipy sparse matrix of size 1*n
#   r is a scalar
class QuadraticFunction:
    def __init__(self, P, q, r, relop=None):
        self.P, self.q, self.r = P, q, r
        self.qarray = np.squeeze(np.asarray(q.todense()))
        self.relop = relop

    # Evalutes f with a numpy array x.
    def eval(self, x):
        return (self.P.dot(x) + self.qarray).dot(x) + self.r

    # Evaluates f with a cvx expression object x.
    def eval_cvx(self, x):
        return cvx.quad_form(x, self.P) + self.q.T*x + self.r

    def violation(self, x):
        assert self.relop is not None
        if self.relop == '==':
            ret = abs(self.eval(x))
        else:
            ret = max(0., self.eval(x))
        return ret

    # Returns the "homogeneous form" matrix M of the function
    # so that (x, 1)^T M (x, 1) is same as f(x).
    def homogeneous_form(self):
        return sp.bmat([[self.P, self.q/2], [self.q.T/2, self.r]])

    # Returns QuadraticFunction f1, f2 such that
    # f(x) = f1(x) - f2(x), with f1 and f2 both convex.
    # Affine and constant components are always put into f1.
    def dc_split(self, use_eigen_split=False):
        n = self.P.shape[0]

        if self.P.nnz == 0: # P is zero
            P1, P2 = sp.csr_matrix((n, n)), sp.csr_matrix((n, n))
        if use_eigen_split:
            lmb, Q = LA.eigh(self.P.todense())
            P1 = sum([Q[:, i]*lmb[i]*Q[:, i].T for i in range(n) if lmb[i] > 0])
            P2 = sum([-Q[:, i]*lmb[i]*Q[:, i].T for i in range(n) if lmb[i] < 0])
            assert abs(np.sum(P1 - P2 - self.P)) < 1e-8
        else:
            lmb_min = np.min(LA.eigh(self.P.todense())[0])
            if lmb_min < 0:
                P1 = self.P + (1-lmb_min)*sp.identity(n)
                P2 = (1-lmb_min)*sp.identity(n)
            else:
                P1 = self.P
                P2 = sp.csr_matrix((n, n))
        f1 = QuadraticFunction(P1, self.q, self.r)
        f2 = QuadraticFunction(P2, sp.csc_matrix((n, 1)), 0)
        return (f1, f2)

    # Returns the one-variable function when regarding f(x)
    # as a quadratic expression in x[k].
    # f is an instance of QuadraticFunction
    # return value is an instance of OneVarQuadraticFunction
    # TODO: speedup
    def get_onevar_func(self, x, k):
        z = np.copy(x)
        z[k] = 0
        t2 = self.P[k, k]
        t1 = 2*self.P[k, :].dot(z)[0] + self.qarray[k]
        t0 = (self.P.dot(z) + self.qarray).dot(z) + self.r
        return OneVarQuadraticFunction(t2, t1, t0, self.relop)

class OneVarQuadraticFunction(QuadraticFunction):
    def __init__(self, P, q, r, relop=None):
        self.P, self.q, self.r = P, q, r
        self.relop = relop

    def __repr__(self):
        return '%+.3f x^2 %+.3f x %+.3f' % (self.P, self.q, self.r)

    def eval(self, x):
        if np.isinf(x):
            if self.P != 0: return self.P*x*x
            if self.q != 0: return self.q*x
            return r
        return x*(self.P*x + self.q) + self.r

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
    def better(self, x1, x2, tol=1e-6): # returns the better point
        v1 = max(self.violations(x1))
        v2 = max(self.violations(x2))
        f1 = self.f0.eval(x1)
        f2 = self.f0.eval(x2)
        if v1 < v2 - tol: return x1
        if v2 < v1 - tol: return x2
        if f1 < f2: return x1
        return x2

# TODO: optimize repeated calculations (cache factors, etc.)
def onecons_qcqp(z, f, tol=1e-6):
    """ Solves a nonconvex problem
      minimize ||x-z||_2^2
      subject to f(x) = x^T P x + q^T x + r ~ 0
      where the relation ~ is given by f.relop (either <= or ==)
    """

    # if constraint is ineq and z is feasible: z is the solution
    if f.relop == '<=' and f.eval(z) <= 0:
        return z

    Psymm = (f.P + f.P.T)/2.
    lmb, Q = LA.eigh(np.asarray(Psymm.todense()))
    zhat = Q.T.dot(z)
    qhat = Q.T.dot(f.qarray)

    # now solve a transformed problem
    # minimize ||xhat - zhat||_2^2
    # subject to sum(lmb_i xhat_i^2) + qhat^T xhat + r = 0
    # constraint is now equality from
    # complementary slackness
    xhat = lambda nu: -np.divide(nu*qhat-2*zhat, 2*(1+nu*lmb))
    phi = lambda xhat: lmb.dot(np.power(xhat, 2)) + qhat.dot(xhat) + f.r

    s = -np.inf
    e = np.inf
    for l in lmb:
        if l > 0: s = max(s, -1./l)
        if l < 0: e = min(e, -1./l)
    if s == -np.inf:
        s = -1.
        while phi(xhat(s)) <= 0: s *= 2.
    if e == np.inf:
        e = 1.
        while phi(xhat(e)) >= 0: e *= 2.
    while e-s > tol:
        m = (s+e)/2.
        p = phi(xhat(m))
        if p > 0: s = m
        elif p < 0: e = m
        else:
            s = e = m
            break
    nu = (s+e)/2.
    return Q.dot(xhat(nu))

def get_feasible_intervals(f, s=0, tol=1e-4):
    p, q, r = f.P, f.q, f.r
    if f.relop == '==': # |px^2 + qx + r| <= s
        f1 = OneVarQuadraticFunction(p, q, r-s, '<=')
        f2 = OneVarQuadraticFunction(-p, -q, -r-s, '<=')
        I = []
        for I1 in get_feasible_intervals(f1):
            for I2 in get_feasible_intervals(f2):
                i = (max(I1[0], I2[0]), min(I1[1], I2[1]))
                if i[0] <= i[1]:
                    I.append(i)
    else: # px^2 + qx + r-s <= 0
        if p > tol:
            D = q*q - 4*p*(r-s)
            if D >= 0:
                rD = np.sqrt(D)
                I = [((-q-rD)/(2*p), (-q+rD)/(2*p))]
            else: # never feasible
                I = []
        elif p < -tol:
            D = q*q - 4*p*(r-s)
            if D >= 0:
                rD = np.sqrt(D)
                # note that p < 0
                I = [(-np.inf, (q+rD)/(2*p)), ((q-rD)/(2*p), np.inf)]
            else: # always feasible
                I = [(-np.inf, np.inf)]
        else:
            if q > tol:
                I = [(-np.inf, (s-r)/q)]
            elif q < -tol:
                I = [((s-r)/q, np.inf)]
            else: # always feasible
                I = [(-np.inf, np.inf)]
    return I


# returns the optimal point of the following program, or None if infeasible
#   minimize f0(x)
#   subject to fi(x) ~ s
# where the only variable is a real number x
# The relation operator ~ can be <= or ==. In case ~ is ==,
# the constraint means |fi(x)| <= s.
def onevar_qcqp(f0, fs, s, tol=1e-4):
    # O(m log m) routine for finding feasible set
    Is = list(chain(*[get_feasible_intervals(f, s) for f in fs]))
    m = len(fs)
    counts = defaultdict(lambda: 0, {-np.inf: +1, +np.inf: -1})
    for I in Is:
        counts[I[0]] += 1
        counts[I[1]] -= 1
    xs = [x for x in sorted(counts.items()) if x[1] != 0]
    C = []
    tot = 0
    for i in range(len(xs)):
        tot += xs[i][1]
        if tot == m and xs[i][1] == -1:
            C.append((xs[i-1][0], xs[i][0]))

    # no feasible points
    if len(C) == 0: return None

    bestxs = []
    bestf = np.inf
    p, q = f0.P, f0.q

    # any point in C works
    # not using tolerance to check zeroness
    if p == 0 and q == 0:
        return np.random.uniform(*C[np.random.choice(len(C))])

    # unconstrained minimizer
    x0 = -q/(2.*p) if p > 0 else np.nan
    # endpoints of feasible intervals
    for I in C:
        if I[0] <= x0 and x0 <= I[1]: return x0
        # note that endpoints or the function values can be +-inf
        (fl, fr) = (f0.eval(I[0]), f0.eval(I[1]))
        if bestf > fl:
            (bestxs, bestf) = [I[0]], fl
        elif bestf == fl:
            bestxs.append(I[0])
        if bestf > fr:
            (bestxs, bestf) = [I[1]], fr
        elif bestf == fr:
            bestxs.append(I[1])

    if len(bestxs) == 0:
        return None
    else:
        return np.random.choice(bestxs)
