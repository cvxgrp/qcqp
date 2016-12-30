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
import warnings
import cvxpy as cvx
import numpy as np
from numpy import linalg as LA
import cvxpy.lin_ops.lin_utils as lu
import scipy.sparse as sp
import scipy.sparse.linalg as SLA
from cvxpy.utilities import QuadCoeffExtractor
from joblib import Parallel, delayed

def get_id_map(vars):
    id_map = {}
    N = 0
    for x in vars:
        id_map[x.id] = N
        N += x.size[0]*x.size[1]
    return id_map, N

def check_quadraticity(prob):
    # check quadraticity
    if not prob.objective.args[0].is_quadratic():
        raise Exception("Objective is not quadratic.")
    if not all([constr._expr.is_quadratic() for constr in prob.constraints]):
        raise Exception("Not all constraints are quadratic.")
    if prob.is_dcp():
        warnings.warn("Problem is already convex; specifying solve method is unnecessary.")

def get_quadratic_forms(prob):
    """Returns the coefficient matrices and variable-index map
    """
    id_map, N = get_id_map(prob.variables())
    extractor = QuadCoeffExtractor(id_map, N)

    P0, q0, r0 = extractor.get_coeffs(prob.objective.args[0])
    q0 = q0.T
    P0 = P0[0]
    if prob.objective.NAME == "maximize":
        P0 = -P0
        q0 = -q0
        r0 = -r0

    Ps = []
    qs = []
    rs = []
    relops = []
    for constr in prob.constraints:
        sz = constr._expr.size[0]*constr._expr.size[1]
        Pc, qc, rc = extractor.get_coeffs(constr._expr)
        for i in range(sz):
            Ps.append(Pc[i])
            qs.append(qc[i, :].T)
            rs.append(rc[i])
            relops.append(constr.OP_NAME)

    return (P0, q0, r0, Ps, qs, rs, relops, id_map, N)

def solve_relaxation(N, P0, q0, r0, Ps, qs, rs, relops, *args, **kwargs):
    """Solve the SDP relaxation.
    """

    # lifted variables and semidefinite constraint
    X = cvx.Semidef(N + 1)

    W = sp.bmat([[P0, q0/2], [q0.T/2, r0]])
    rel_obj = cvx.Minimize(cvx.sum_entries(cvx.mul_elemwise(W, X)))
    rel_constr = [X[N, N] == 1]

    for i in range(len(Ps)):
        W = sp.bmat([[Ps[i], qs[i]/2], [qs[i].T/2, rs[i]]])
        lhs = cvx.sum_entries(cvx.mul_elemwise(W, X))
        if relops[i] == '==':
            rel_constr.append(lhs == 0)
        else:
            rel_constr.append(lhs <= 0)

    rel_prob = cvx.Problem(rel_obj, rel_constr)
    rel_prob.solve(*args, **kwargs)

    if rel_prob.status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
        print ("Relaxation problem status: " + rel_prob.status)
        return None, rel_prob.value

    return X.value, rel_prob.value

def generate_samples(use_sdp, num_samples, N, P0, q0, r0, Ps, qs, rs, relops, *args, **kwargs):
    if use_sdp:
        X, _ = solve_relaxation(N, P0, q0, r0, Ps, qs, rs, relops, *args, **kwargs)
        mu = np.asarray(X[:-1, -1]).flatten()
        Sigma = X[:-1, :-1] - mu*mu.T
        samples = np.random.multivariate_normal(mu, Sigma, num_samples)
    else:
        samples = np.random.randn(num_samples, N)
    ret = [np.asmatrix(samples[i, :].reshape((N, 1))) for i in range(num_samples)]
    return ret

# TODO: optimize repeated calculations (cache factors, etc.)
def one_qcqp(z, A, b, c, relop='<=', tol=1e-6):
    """ Solves a nonconvex problem
      minimize ||x-z||_2^2
      subject to x^T A x + b^T x + c ~ 0
      where the relation ~ is given by relop
    """

    # if constraint is ineq and z is feasible: z is the solution
    if relop == '<=' and z.T*A*z + z.T*b + c <= 0:
        return z

    lmb, Q = map(np.asmatrix, LA.eigh(A.todense()))
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
    while e-s > tol:
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

def assign_vars(xs, vals):
    ind = 0
    for x in xs:
        size = x.size[0]*x.size[1]
        x.value = np.reshape(vals[ind:ind+size], x.size, order='F')
        ind += size

def sdp_relax(self, *args, **kwargs):
    check_quadraticity(self)
    P0, q0, r0, Ps, qs, rs, relops, id_map, N = get_quadratic_forms(self)
    X, sdp_bound = solve_relaxation(N, P0, q0, r0, Ps, qs, rs, relops, *args, **kwargs)
    if self.objective.NAME == "maximize":
        sdp_bound = -sdp_bound
    assign_vars(self.variables(), X[:, -1])
    return sdp_bound

def x_update(x, y, z, rho, P, q, r, relop):
    return one_qcqp(z + (1/rho)*y, P, q, r, relop)

def y_update(x, y, z, rho):
    return y + rho*(z - x)

def qcqp_admm(self, use_sdp=True,
    num_samples=100, num_iters=1000, viollim=1e10,
    tol=1e-4, *args, **kwargs):
    check_quadraticity(self)
    P0, q0, r0, Ps, qs, rs, relops, id_map, N = get_quadratic_forms(self)

    M = len(Ps)

    lmb0, P0Q = map(np.asmatrix, LA.eigh(P0.todense()))
    lmb_min = np.min(lmb0)
    if lmb_min < 0: rho = 2*(1-lmb_min)/M
    else: rho = 1./M
    rho *= 5

    bestx = None
    bestf = np.inf

    samples = generate_samples(use_sdp, num_samples, N, P0, q0, r0, Ps, qs, rs, relops, *args, **kwargs)

    for x0 in samples:
        z = x0
        xs = [x0 for i in range(M)]
        ys = [np.zeros((N, 1)) for i in range(M)]
        print("trial %d: %f" % (sample, bestf))

        zlhs = 2*P0 + rho*M*sp.identity(N)
        lstza = None
        for t in range(num_iters):
            rhs = sum([rho*xs[i]-ys[i] for i in range(M)]) - q0
            z = np.asmatrix(SLA.spsolve(zlhs.tocsr(), rhs)).T
            xs = Parallel(n_jobs=4)(
                delayed(x_update)(xs[i], ys[i], z, rho, Ps[i], qs[i], rs[i], relops[i])
                for i in range(M)
            )
            ys = Parallel(n_jobs=4)(
                delayed(y_update)(xs[i], ys[i], z, rho)
                for i in range(M)
            )

            za = (sum(xs)+z)/(M+1)
            #if lstza is not None and LA.norm(lstza-za) < tol:
            #    break
            lstza = za
            maxviol = 0
            for i in range(M):
                (P1, q1, r1) = (Ps[i], qs[i], rs[i])
                viol = (za.T*P1*za + za.T*q1 + r1)[0, 0]
                if relops[i] == '==': viol = abs(viol)
                else: viol = max(0, viol)
                maxviol = max(maxviol, viol)

            #print(t, maxviol)

            objt = (za.T*P0*za + za.T*q0 + r0)[0, 0]
            if maxviol > viollim:
                rho *= 2
                break

            if maxviol < tol and bestf > objt:
                bestf = objt
                bestx = za
                print("best found point has objective: %.5f" % (bestf))
                print("best found point: ", bestx)


    #print("Iteration %d:" % (t))
    print("best found point has objective: %.5f" % (bestf))
    print("best found point: ", bestx)

    assign_vars(self.variables(), bestx)
    return bestf

# given indefinite P
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


def qcqp_dccp(self, use_sdp=True, use_eigen_split=False,
    num_samples=100, *args, **kwargs):
    check_quadraticity(self)
    try:
        import dccp
    except ImportError:
        print("DCCP package is not installed; qcqp-dccp method is unavailable.")
        raise

    P0, q0, r0, Ps, qs, rs, relops, id_map, N = get_quadratic_forms(self)

    M = len(Ps)

    x = cvx.Variable(N)
    T = cvx.Variable() # objective function

    obj = cvx.Minimize(T)
    P0p, P0m = split_quadratic(P0, use_eigen_split)
    cons = [cvx.quad_form(x, P0p)+q0.T*x+r0 <= cvx.quad_form(x, P0m)+T]

    for i in range(M):
        Pp, Pm = split_quadratic(Ps[i], use_eigen_split)
        lhs = cvx.quad_form(x, Pp)+qs[i].T*x+rs[i]
        rhs = cvx.quad_form(x, Pm)
        if relops[i] == '==':
            cons.append(lhs == rhs)
        else:
            cons.append(lhs <= rhs)

    samples = generate_samples(use_sdp, num_samples, N, P0, q0, r0, Ps, qs, rs, relops, *args, **kwargs)

    prob = cvx.Problem(obj, cons)
    bestx = None
    bestf = np.inf
    for x0 in samples:
        x.value = x0
        val = prob.solve(method='dccp')[0]
        if val is not None and bestf > val:
            bestf = val
            bestx = x.value
            print("found new point with f: %.5f" % (bestf))

    if self.objective.NAME == "maximize":
        bestf = -bestf
    assign_vars(self.variables(), x.value)
    return bestf

def get_violation(x, Ps, qs, rs, relops):
    M = len(Ps)
    ret = []
    for i in range(M):
        v = x.T@Ps[i]@x + x.T@qs[i] + rs[i]
        if relops[i] == '==':
            ret.append(abs(v))
        else:
            ret.append(max(0, v))
    return ret

def get_violation_onevar(x, coefs):
    ret = []
    for c in coefs:
        p, q, r = c
        ret.append(max(0, p*x**2 + q*x + r))
    return ret

# given interval I and array of intervals C = [I1, I2, ..., Im]
# returns [I1 cap I, I2 cap I, ..., Im cap I]
def interval_intersection(C, I):
    ret = []
    for J in C:
        IJ = (max(I[0], J[0]), min(I[1], J[1]))
        if IJ[0] <= IJ[1]:
            ret.append(IJ)
    return ret

# TODO: fix relops behaviors
# coefs = [(p0, q0, r0), (p1, q1, r1), ..., (pm, qm, rm)]
# returns the optimal point of the following program, or None if infeasible
#   minimize p0 x^2 + q0 x + r0
#   subject to pi x^2 + qi x + ri <= s
# TODO: more efficient method for computing C
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
                I = (-np.inf, (-q-rD)/(2*p))
                C1 = interval_intersection(C, I)
                I = ((-q+rD)/(2*p), np.inf)
                C2 = interval_intersection(C, I)
                C = C1 + C2
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

# regard x^T P x + q^T x + r as a quadratic expression in xi
# and returns the coefficients
def get_onevar_coeffs(i, P, q, r):
    t2 = P[i, i]
    t1 = 2*(P[i, :].sum() - P[i, i]) + q[i, 0]
    t0 = P[:i, :i].sum() + 2*P[:i, i+1:].sum() \
         + P[i+1:, i+1:].sum() + r
    return (t2, t1, t0)

# rewrite the dirty stuff below
def coord_descent(self, use_sdp=True,
    num_samples=100, num_iters=1000,
    bsearch_tol=1e-4, tol=1e-4, *args, **kwargs):
    check_quadraticity(self)
    P0, q0, r0, Ps, qs, rs, relops, id_map, N = get_quadratic_forms(self)

    M = len(Ps)

    bestx = None
    bestf = np.inf

    samples = generate_samples(use_sdp, num_samples, N, P0, q0, r0, Ps, qs, rs, relops, *args, **kwargs)

    # debugging purpose
    counter = 0
    for x in samples:
        # TODO: give some epsilon slack to equality constraint
        # number of iterations since last infeasibility improvement
        update_counter = 0
        # phase 1: optimize infeasibility
        failed = False
        while True:
            # optimize over x[i]
            for i in range(N):
                coefs = [(0, 0, 0)]
                for j in range(M):
                    # quadratic, linear, constant terms
                    c = get_onevar_coeffs(i, Ps[j], qs[j], rs[j])
                    # constraint not relevant to xi is ignored
                    if abs(c[0]) > tol or abs(c[1]) > tol:
                        if relops[j] == '<=':
                            coefs.append(c)
                        else:
                            coefs.append(c)
                            coefs.append((-c[0], -c[1], -c[2]))

                viol = max(get_violation_onevar(x[i], coefs))
                new_xi = x[i]
                new_viol = viol
                ss, es = 0, viol
                while es - ss > bsearch_tol:
                    s = (ss + es) / 2
                    xi = onevar_qcqp(coefs, s, tol)
                    if xi is None:
                        ss = s
                    else:
                        new_xi = xi
                        new_viol = s
                        es = s
                if new_viol < viol:
                    x[i] = new_xi
                    update_counter = 0
                else:
                    update_counter += 1
                    if update_counter == N:
                        failed = True
                        break
            if failed: break
            viol = max(get_violation(x, Ps, qs, rs, relops))
            if viol < tol: break

        # phase 2: optimize objective over feasible points
        if failed: continue

        update_counter = 0
        converged = False
        for t in range(num_iters):
            # optimize over x[i]
            for i in range(N):
                coefs = [get_onevar_coeffs(i, P0, q0, r0)]
                for j in range(M):
                    # quadratic, linear, constant terms
                    c = get_onevar_coeffs(i, Ps[j], qs[j], rs[j])
                    # constraint not relevant to xi is ignored
                    if abs(c[0]) > tol or abs(c[1]) > tol:
                        if relops[j] == '<=':
                            coefs.append(c)
                        else:
                            coefs.append(c)
                            coefs.append((-c[0], -c[1], -c[2]))
                new_xi = onevar_qcqp(coefs, 0, tol)
                if np.abs(new_xi - x[i]) > tol:
                    x[i] = new_xi
                    update_counter = 0
                else:
                    update_counter += 1
                    if update_counter == N:
                        converged = True
                        break
            if converged: break
        if bestf > x.T@P0@x + q0.T@x + r0:
            bestf = x.T@P0@x + q0.T@x + r0
            bestx = x
            print("best found point has objective: %.5f" % (bestf))
            #print("best found point: ", bestx)

        counter += 1
        print("trial %d: %f" % (counter, bestf))


    print("best found point has objective: %.5f" % (bestf))
    #print("best found point: ", bestx)

    assign_vars(self.variables(), bestx)
    return bestf


# Add solution methods to Problem class.
cvx.Problem.register_solve("sdp-relax", sdp_relax)
# TODO: have a better structure for the repeated lines
# in the solve methods
cvx.Problem.register_solve("qcqp-admm", qcqp_admm)
cvx.Problem.register_solve("qcqp-dccp", qcqp_dccp)
cvx.Problem.register_solve("coord-descent", coord_descent)
