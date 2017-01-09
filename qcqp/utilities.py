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

# given interval I and array of intervals C = [I1, I2, ..., Im]
# returns [I1 cap I, I2 cap I, ..., Im cap I]
def interval_intersection(C, I):
    ret = []
    for J in C:
        IJ = (max(I[0], J[0]), min(I[1], J[1]))
        if IJ[0] <= IJ[1]:
            ret.append(IJ)
    return ret

# coefs = [(p0, q0, r0), (p1, q1, r1), ..., (pm, qm, rm)]
# returns the optimal point of the following program, or None if infeasible
#   minimize p0 x^2 + q0 x + r0
#   subject to pi x^2 + qi x + ri <= s
# TODO: efficiently find feasible set using BST
def onevar_qcqp(coefs, s, tol=1e-4):
    # feasible set as a collection of disjoint intervals
    C = [(-np.inf, np.inf)]
    for cons in coefs[1:]:
        (p, q, r) = cons
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

