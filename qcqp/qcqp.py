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
import cvxpy as cvx
import numpy as np
import cvxpy.lin_ops.lin_utils as lu
import scipy.sparse as sp
from numpy import linalg as LA
import scipy.sparse.linalg as SLA
from cvxpy.utilities import QuadCoeffExtractor
from utilities import *
import logging
import settings as s

logging.basicConfig(filename='qcqp.log', filemode='w', level=logging.INFO)

def get_id_map(xs):
    id_map = {}
    N = 0
    for x in xs:
        id_map[x.id] = N
        N += x.size[0]*x.size[1]
    return id_map, N


def get_qcqp_form(prob):
    """Returns the problem metadata in QCQP class
    """
    # Check quadraticity
    if not prob.objective.args[0].is_quadratic():
        raise Exception("Objective is not quadratic.")
    if not all([constr._expr.is_quadratic() for constr in prob.constraints]):
        raise Exception("Not all constraints are quadratic.")
    if prob.is_dcp():
        logging.warning("Problem is already convex; specifying solve method is unnecessary.")

    extractor = QuadCoeffExtractor(*get_id_map(prob.variables()))

    P0, q0, r0 = extractor.get_coeffs(prob.objective.args[0])
    # unpacking values
    P0, q0, r0 = (P0[0]+P0[0].T)/2., q0.T.tocsc(), r0[0]

    if prob.objective.NAME == "maximize":
        P0, q0, r0 = -P0, -q0, -r0

    f0 = QuadraticFunction(P0, q0, r0)

    fs = []
    for constr in prob.constraints:
        sz = constr._expr.size[0]*constr._expr.size[1]
        Pc, qc, rc = extractor.get_coeffs(constr._expr)
        for i in range(sz):
            fs.append(QuadraticFunction((Pc[i]+Pc[i].T)/2., qc[i, :].T.tocsc(), rc[i], constr.OP_NAME))

    return QCQP(f0, fs)

def assign_vars(xs, vals):
    if vals is None:
        for x in xs:
            size = x.size[0]*x.size[1]
            x.value = np.full(x.size, np.nan)
    else:
        ind = 0
        for x in xs:
            size = x.size[0]*x.size[1]
            x.value = np.reshape(vals[ind:ind+size], x.size, order='F')
            ind += size

def flatten_vars(xs, n):
    ret = np.empty(n)
    ind = 0
    for x in xs:
        size = x.size[0]*x.size[1]
        ret[ind:ind+size] = np.ravel(x.value, order='F')
    return ret

def solve_relaxation(prob, *args, **kwargs):
    """Solve the SDP relaxation.
    """

    # lifted variables and semidefinite constraint
    X = cvx.Semidef(prob.n + 1)

    W = prob.f0.homogeneous_form()
    rel_obj = cvx.Minimize(cvx.sum_entries(cvx.mul_elemwise(W, X)))
    rel_constr = [X[-1, -1] == 1]

    for f in prob.fs:
        W = f.homogeneous_form()
        lhs = cvx.sum_entries(cvx.mul_elemwise(W, X))
        if f.relop == '==':
            rel_constr.append(lhs == 0)
        else:
            rel_constr.append(lhs <= 0)

    rel_prob = cvx.Problem(rel_obj, rel_constr)
    rel_prob.solve(*args, **kwargs)

    if rel_prob.status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
        raise Exception("Relaxation problem status: %s" % rel_prob.status)

    return X.value, rel_prob.value


# TODO: rewrite the dirty stuff below
def coord_descent(x, prob, *args, **kwargs):
    num_iters = kwargs.get('num_iters', 1000)
    bsearch_tol = kwargs.get('bsearch_tol', 1e-4)
    viol_tol = kwargs.get('viol_tol', 1e-3)
    tol = kwargs.get('tol', 1e-4)

    # number of iterations since last infeasibility improvement
    update_counter = 0
    # phase 1: optimize infeasibility
    failed = False
    logging.info("Phase 1 starts")
    # TODO: correct termination condition with tolerance
    viol_last = np.inf
    while viol_last > viol_tol:
        # optimize over x[i]
        for i in range(prob.n):
            obj = OneVarQuadraticFunction(0, 0, 0)
            nfs = [f.get_onevar_func(x, i) for f in prob.fs]
            # TODO: maybe this shouldn't be here?
            nfs = [f for f in nfs if abs(f.P) > tol or abs(f.q) > tol]
            viol = max([f.violation(x[i]) for f in nfs])
            logging.debug("Current violation in x[%d]: %.3f", i, viol)
            logging.debug("Current point: %s", x)
            new_xi = x[i]
            new_viol = viol
            ss, es = 0, viol
            while es - ss > bsearch_tol:
                s = (ss + es) / 2
                xi = onevar_qcqp(obj, nfs, s)
                if xi is None:
                    ss = s
                else:
                    new_xi = xi
                    new_viol = s
                    es = s
            if new_viol < viol:
                x[i] = new_xi
                update_counter = 0
                logging.debug("Violation reduction %.3f -> %.3f", viol, new_viol)
            else:
                update_counter += 1
                if update_counter == prob.n:
                    failed = True
                    break
        #if failed: break
        viol = max(prob.violations(x))
        logging.info("Maximum violation: %.6f -> %.6f", viol_last, viol)
        if viol_last <= viol + bsearch_tol:
            break
        viol_last = viol

    # TODO: find correct termination condition with tolerance
    # phase 2: optimize objective over feasible points
    #if failed: continue
    if viol > viol_tol: return x
    logging.info("Phase 2 starts")
    fval = prob.f0.eval(x)
    update_counter = 0
    converged = False
    for t in range(num_iters):
        # optimize over x[i]
        for i in range(prob.n):
            obj = prob.f0.get_onevar_func(x, i)
            nfs = [f.get_onevar_func(x, i) for f in prob.fs]
            # TODO: maybe this shouldn't be here?
            nfs = [f for f in nfs if abs(f.P) > tol or abs(f.q) > tol]
            new_xi = onevar_qcqp(obj, nfs, tol)
            if new_xi is not None and np.abs(new_xi - x[i]) > tol:
                x[i] = new_xi
                update_counter = 0
            else:
                update_counter += 1
                if update_counter == prob.n:
                    converged = True
                    break
        if converged: break

    return x


def admm_phase1(prob, x0, tol=1e-4, num_iters=100):
    z = np.copy(x0)
    xs = [np.copy(x0) for i in range(prob.m)]
    us = [np.zeros(prob.n) for i in range(prob.m)]

    for t in range(num_iters):
        z = (sum(xs)-sum(us))/prob.m
        for i in range(prob.m):
            x, u, f = xs[i], us[i], prob.fi(i)
            xs[i] = onecons_qcqp(z + u, f)
        for i in range(prob.m):
            us[i] += z - xs[i]
        if max(prob.violations(z)) < tol:
            break

    return z

def qcqp_admm(x0, prob, *args, **kwargs):
    num_iters = kwargs.get('num_iters', 100)
    viollim = kwargs.get('viollim', 1e4)
    tol = kwargs.get('tol', 5e-2)
    rho = kwargs.get('rho', None)
    # TODO: find a reasonable auto parameter
    if rho is not None:
        lmb0, P0Q = map(np.asmatrix, LA.eigh(prob.f0.P.todense()))
        lmb_min = np.min(lmb0)
        if lmb_min + prob.m*rho < 0:
            logging.error("rho parameter is too small, z-update not convex.")
            logging.error("Minimum possible value of rho: %.3f\n", -lmb_min/prob.m)
            logging.error("Given value of rho: %.3f\n", rho)
            raise

    if rho is None:
        lmb0, P0Q = map(np.asmatrix, LA.eigh(prob.f0.P.todense()))
        lmb_min = np.min(lmb0)
        lmb_max = np.max(lmb0)
        if lmb_min < 0: rho = 2.*(1.-lmb_min)/prob.m
        else: rho = 1./prob.m
        rho *= 50.
        logging.warning("Automatically setting rho to %.3f", rho)

    logging.info("Starting phase 1")
    x0 = prob.better(x0, admm_phase1(prob, x0, tol, num_iters))
    bestx = np.copy(x0)

    z = x0
    xs = [np.copy(x0) for i in range(prob.m)]
    us = [np.zeros(prob.n) for i in range(prob.m)]

    zlhs = 2*(prob.f0.P + rho*prob.m*sp.identity(prob.n))
    last_z = None
    logging.info("Starting phase 2, rho %.3f", rho)
    for t in range(num_iters):
        rhs = 2*rho*(sum(xs)-sum(us)) - prob.f0.qarray
        z = SLA.spsolve(zlhs.tocsr(), rhs)

        # TODO: parallel x/u-updates
        for i in range(prob.m):
            xs[i] = onecons_qcqp(z + us[i], prob.fi(i))
        for i in range(prob.m):
            us[i] += z - xs[i]

        # TODO: termination condition
        if last_z is not None and LA.norm(last_z - z) < tol:
            break
        last_z = z

        maxviol = max(prob.violations(z))
        logging.info("Iteration %d, violation %.3f", t, maxviol)

        if maxviol > viollim: break
        bestx = np.copy(prob.better(z, bestx))

    return bestx


def qcqp_dccp(x0, prob, *args, **kwargs):
    try:
        import dccp
    except ImportError:
        logging.error("DCCP package is not installed; qcqp-dccp method is unavailable.")
        raise

    use_eigen_split = kwargs.get('use_eigen_split', False)
    tau = kwargs.get('tau', 0.005)

    x = cvx.Variable(prob.n)
    x.value = x0
    # dummy objective
    T = cvx.Variable()
    T.value = prob.f0.eval(x0)

    obj = cvx.Minimize(T)
    f0p, f0m = prob.f0.dc_split(use_eigen_split)
    cons = [f0p.eval_cvx(x) <= f0m.eval_cvx(x) + T]

    for f in prob.fs:
        fp, fm = f.dc_split(use_eigen_split)
        if f.relop == '==':
            cons.append(fp.eval_cvx(x) == fm.eval_cvx(x))
        else:
            cons.append(fp.eval_cvx(x) <= fm.eval_cvx(x))

    dccp_prob = cvx.Problem(obj, cons)
    bestx = np.copy(x0)
    try:
        result = dccp_prob.solve(method='dccp', tau=tau)
        if dccp_prob.status == "Converged":
            bestx = prob.better(bestx, np.asarray(x.value).flatten())
    except cvx.error.SolverError:
        pass
    return bestx



class QCQPWrapper:
    def __init__(self, prob):
        self.prob = prob
        self.qcqp_form = get_qcqp_form(prob)
        self.n = self.qcqp_form.n
        self.sdp_sol = None
        self.sdp_bound = None
        self.maximize_flag = (prob.objective.NAME == "maximize")

    def suggest(self, sdp=False, eps=1e-8, *args, **kwargs):
        if sdp and self.sdp_sol is None:
            self.sdp_sol, self.sdp_bound = solve_relaxation(self.qcqp_form, *args, **kwargs)
            if self.maximize_flag:
                self.sdp_bound = -self.sdp_bound
            self.mu = np.asarray(self.sdp_sol[:-1, -1]).flatten()
            self.Sigma = self.sdp_sol[:-1, :-1] - self.mu*self.mu.T + eps*sp.identity(self.n)
        if sdp:
            x = np.random.multivariate_normal(self.mu, self.Sigma, 1)
        else:
            x = np.random.randn(1, self.n)
        x = x[0, :].flatten()
        assign_vars(self.prob.variables(), x)
        f0 = self.qcqp_form.f0.eval(x)
        if self.maximize_flag: f0 = -f0
        return (f0, max(self.qcqp_form.violations(x)))

    def improve(self, method, *args, **kwargs):
        if method not in s.available_methods:
            logging.error("Unknown method: %s\n", method)
            raise
        for x in self.prob.variables():
            if x.value is None:
                self.suggest()
                break
        x0 = flatten_vars(self.prob.variables(), self.n)
        if method == s.COORD_DESCENT:
            x = coord_descent(x0, self.qcqp_form, args, kwargs)
        elif method == s.ADMM:
            x = qcqp_admm(x0, self.qcqp_form, args, kwargs)
        elif method == s.DCCP:
            x = qcqp_dccp(x0, self.qcqp_form, args, kwargs)
        assign_vars(self.prob.variables(), x)
        f0 = self.qcqp_form.f0.eval(x)
        if self.maximize_flag: f0 = -f0
        return (f0, max(self.qcqp_form.violations(x)))
