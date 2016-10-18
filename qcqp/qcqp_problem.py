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

def solve_SDP_relaxation(prob, *args, **kwargs):
    """Solve the SDP relaxation.
    """
    id_map, N = get_id_map(prob.variables())
    extractor = QuadCoeffExtractor(id_map, N)

    # lifted variables and semidefinite constraint
    X = cvx.Semidef(N + 1)

    (Ps, Q, R) = extractor.get_coeffs(prob.objective.args[0])
    M = sp.bmat([[Ps[0], Q.T/2], [Q/2, R]])
    rel_obj = type(prob.objective)(cvx.sum_entries(cvx.mul_elemwise(M, X)))
    rel_constr = [X[N, N] == 1]
    for constr in prob.constraints:
        sz = constr._expr.size[0]*constr._expr.size[1]
        (Ps, Q, R) = extractor.get_coeffs(constr._expr)
        for i in range(sz):
            M = sp.bmat([[Ps[i], Q[i, :].T/2], [Q[i, :]/2, R[i]]])
            c = cvx.sum_entries(cvx.mul_elemwise(M, X))
            if constr.OP_NAME == '==':
                rel_constr.append(c == 0)
            else:
                rel_constr.append(c <= 0)

    rel_prob = cvx.Problem(rel_obj, rel_constr)
    rel_prob.solve(*args, **kwargs)

    if rel_prob.status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
        print ("Relaxation problem status: " + rel_prob.status)
        return None, rel_prob.value, id_map, N

    return X.value, rel_prob.value, id_map, N

# Solves a nonconvex problem
# minimize ||x-z||_2^2
# subject to x^T A x + b^T x + c <= 0 or = 0
def one_qcqp(z, A, b, c, equality_cons=False, tol=1e-6):
    # if constraint is ineq and z is feasible: z is the solution
    if not equality_cons and z.T*A*z + z.T*b + c <= 0:
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

def relax_sdp(self, *args, **kwargs):
    check_quadraticity(self)
    X, sdp_bound, id_map, N = solve_SDP_relaxation(self, *args, **kwargs)

    assign_vars(self.variables(), X[:, -1])

    return sdp_bound

def noncvx_admm(self, num_samples=100, num_iters=1000, viollim=1e10,
    eps=1e-3, *args, **kwargs):
    check_quadraticity(self)

    id_map, N = get_id_map(self.variables())
    extractor = QuadCoeffExtractor(id_map, N)

    (P0, Q0, R0) = extractor.get_coeffs(self.objective.args[0])
    Q0 = Q0.T
    P0 = P0[0]

    if self.objective.NAME == "maximize":
        P0 = -P0
        Q0 = -Q0
        R0 = -R0

    PP = []
    QQ = []
    RR = []
    rel = []
    for constr in self.constraints:
        sz = constr._expr.size[0]*constr._expr.size[1]
        (Ps, Q, R) = extractor.get_coeffs(constr._expr)
        for i in range(sz):
            PP.append(Ps[i])
            QQ.append(Q[i, :].T)
            RR.append(R[i])
            rel.append(constr.OP_NAME == '==')

    M = len(PP)

    lmb0, P0Q = map(np.asmatrix, LA.eigh(P0.todense()))
    lmb_min = np.min(lmb0)
    if lmb_min < 0: rho = 2*(1-lmb_min)/M
    else: rho = 1./M
    rho *= 5

    bestx = None
    bestf = np.inf

    #X, sdp_bound, id_map, N = solve_SDP_relaxation(self, *args, **kwargs)

    # TODO
    # (1) generate random samples using X
    # (2) optimize repeated calculations (cache factors, etc.)
    for sample in range(num_samples):
        print("trial %d: %f" % (sample, bestf))
        z = np.asmatrix(np.random.randn(N, 1))
        xs = np.asmatrix(np.random.randn(N, M))
        for i in range(M): xs[:, i] = z
        ys = np.asmatrix(np.zeros((N, M)))

        zlhs = 2*P0 + rho*M*sp.identity(N)
        lstza = None
        for t in range(num_iters):
            rhs = np.sum(rho*xs-ys, 1) - Q0
            z = np.asmatrix(SLA.spsolve(zlhs.tocsr(), rhs)).T
            for i in range(M):
                zz = z + (1/rho)*ys[:, i]
                xs[:, i] = one_qcqp(zz, PP[i], QQ[i], RR[i], rel[i])
                ys[:, i] += rho*(z - xs[:, i])

            za = (np.sum(xs, 1)+z)/(M+1)
            #if lstza is not None and LA.norm(lstza-za) < eps:
            #    break
            lstza = za
            maxviol = 0
            for i in range(M):
                (P1, q1, r1) = (PP[i], QQ[i], RR[i])
                viol = (za.T*PP[i]*za + za.T*QQ[i] + RR[i])[0, 0]
                if rel[i] == True: viol = abs(viol)
                else: viol = max(0, viol)
                maxviol = max(maxviol, viol)

            #print(t, maxviol)

            objt = (za.T*P0*za + za.T*Q0 + R0)[0, 0]
            if maxviol > viollim:
                rho *= 2
                break

            if maxviol < eps and bestf > objt:
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
def split_quadratic(P):
    n = P.shape[0]
    # zero matrix
    if P.nnz == 0:
        return (sp.csr_matrix((n, n)), sp.csr_matrix((n, n)))
    lmb_min = np.min(LA.eigh(P.todense())[0])
    if lmb_min < 0:
        return (P + (1-lmb_min)*sp.identity(n), (1-lmb_min)*sp.identity(n))
    return (P, sp.csr_matrix((n, n)))


def qcqp_dccp(self, *args, **kwargs):
    check_quadraticity(self)
    try:
        import dccp
    except ImportError:
        print("DCCP package is not installed; qcqp-dccp method is unavailable.")
        raise

    id_map, N = get_id_map(self.variables())
    extractor = QuadCoeffExtractor(id_map, N)

    (P0, Q0, R0) = extractor.get_coeffs(self.objective.args[0])
    Q0 = Q0.T
    P0 = P0[0]

    if self.objective.NAME == "maximize":
        P0 = -P0
        Q0 = -Q0
        R0 = -R0

    PP = []
    QQ = []
    RR = []
    rel = []
    for constr in self.constraints:
        sz = constr._expr.size[0]*constr._expr.size[1]
        (Ps, Q, R) = extractor.get_coeffs(constr._expr)
        for i in range(sz):
            PP.append(Ps[i])
            QQ.append(Q[i, :].T)
            RR.append(R[i])
            rel.append(constr.OP_NAME == '==')

    M = len(PP)

    X = cvx.Variable(N)
    T = cvx.Variable() # objective function

    obj = cvx.Minimize(T)
    (P0p, P0m) = split_quadratic(P0)
    cons = [cvx.quad_form(X, P0p)+X.T*Q0+R0 <= cvx.quad_form(X, P0m)+T]

    for i in range(M):
        (Pp, Pm) = split_quadratic(PP[i])
        cons.append(cvx.quad_form(X, Pp)+X.T*QQ[i]+RR[i] <= cvx.quad_form(X, Pm))

    prob = cvx.Problem(obj, cons)
    #assert prob.is_dccp(), "Unknown error: Failed to form a DCCP problem."

    # TODO: solve SDP relaxation, generate multiple sample points

    result = prob.solve(method='dccp')[0]
    if self.objective.NAME == "maximize":
        result = -result
    assign_vars(self.variables(), X.value)
    return result

# Add solution methods to Problem class.
cvx.Problem.register_solve("relax-SDP", relax_sdp)
cvx.Problem.register_solve("noncvx-admm", noncvx_admm)
cvx.Problem.register_solve("qcqp-dccp", qcqp_dccp)
