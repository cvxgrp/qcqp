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

def noncvx_admm(self, use_sdp=True,
    num_samples=100, num_iters=1000, viollim=1e10,
    tol=1e-3, *args, **kwargs):
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

    if use_sdp:
        X, _ = solve_relaxation(N, P0, q0, r0, Ps, qs, rs, relops, *args, **kwargs)
        mu = np.asarray(X[:-1, -1]).flatten()
        print(mu, mu.shape)
        Sigma = X[:-1, :-1] - mu*mu.T
        samples = np.random.multivariate_normal(mu, Sigma, num_samples)
    else:
        samples = np.random.randn(num_samples, N)

    # each row of samples is a starting point
    for sample in range(num_samples):
        x0 = np.asmatrix(samples[sample, :].reshape((N, 1)))
        z = x0
        xs = np.repeat(x0, M, axis=1)
        ys = np.asmatrix(np.zeros((N, M)))
        print("trial %d: %f" % (sample, bestf))

        zlhs = 2*P0 + rho*M*sp.identity(N)
        lstza = None
        for t in range(num_iters):
            rhs = np.sum(rho*xs-ys, 1) - q0
            z = np.asmatrix(SLA.spsolve(zlhs.tocsr(), rhs)).T
            for i in range(M):
                zz = z + (1/rho)*ys[:, i]
                xs[:, i] = one_qcqp(zz, Ps[i], qs[i], rs[i], relops[i])
                ys[:, i] += rho*(z - xs[:, i])

            za = (np.sum(xs, 1)+z)/(M+1)
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

    if use_sdp:
        X, _ = solve_relaxation(N, P0, q0, r0, Ps, qs, rs, relops, *args, **kwargs)
        mu = np.asarray(X[:-1, -1]).flatten()
        print(mu, mu.shape)
        Sigma = X[:-1, :-1] - mu*mu.T
        samples = np.random.multivariate_normal(mu, Sigma, num_samples)
    else:
        samples = np.random.randn(num_samples, N)

    # each row of samples is a starting point
    prob = cvx.Problem(obj, cons)
    bestx = None
    bestf = np.inf
    for sample in range(num_samples):
        x.value = np.asmatrix(samples[sample, :].reshape((N, 1)))
        val = prob.solve(method='dccp')[0]
        if val is not None and bestf > val:
            bestf = val
            bestx = x.value
            print("found new point with f: %.5f" % (bestf))

    if self.objective.NAME == "maximize":
        bestf = -bestf
    assign_vars(self.variables(), x.value)
    return bestf

# Add solution methods to Problem class.
cvx.Problem.register_solve("sdp-relax", sdp_relax)
cvx.Problem.register_solve("qcqp-admm", noncvx_admm)
cvx.Problem.register_solve("qcqp-dccp", qcqp_dccp)
