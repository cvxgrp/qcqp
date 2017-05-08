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
import cvxpy as cvx
import numpy as np
import cvxpy.lin_ops.lin_utils as lu
import scipy.sparse as sp
from numpy import linalg as LA
import scipy.sparse.linalg as SLA
from utilities import *
import logging
import settings as s

logging.basicConfig(filename='qcqp.log', filemode='w', level=logging.INFO)

def solve_spectral(prob, *args, **kwargs):
    """Solve the spectral relaxation with lambda = 1.
    """

    # TODO: do this efficiently without SDP lifting

    # lifted variables and semidefinite constraint
    X = cvx.Semidef(prob.n + 1)

    W = prob.f0.homogeneous_form()
    rel_obj = cvx.Minimize(cvx.sum_entries(cvx.mul_elemwise(W, X)))

    W1 = sum([f.homogeneous_form() for f in prob.fs if f.relop == '<='])
    W2 = sum([f.homogeneous_form() for f in prob.fs if f.relop == '=='])

    rel_prob = cvx.Problem(
        rel_obj,
        [
            cvx.sum_entries(cvx.mul_elemwise(W1, X)) <= 0,
            cvx.sum_entries(cvx.mul_elemwise(W2, X)) == 0,
            X[-1, -1] == 1
        ]
    )
    rel_prob.solve(*args, **kwargs)

    if rel_prob.status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
        raise Exception("Relaxation problem status: %s" % rel_prob.status)

    (w, v) = LA.eig(X.value)
    return np.sqrt(np.max(w))*np.asarray(v[:-1, np.argmax(w)]).flatten(), rel_prob.value

def solve_sdr(prob, *args, **kwargs):
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


# phase 1: optimize infeasibility
def coord_descent_phase1(x0, prob, num_iters=1000,
    viol_tol=1e-2, tol=1e-4):
    logging.info("Phase 1 starts")
    x = np.copy(x0)
    # number of iterations since last infeasibility improvement
    update_counter = 0
    failed = False
    # TODO: correct termination condition with tolerance
    viol_last = np.inf
    for t in range(num_iters):
        if viol_last < viol_tol: break
        # optimize over x[i]
        for i in range(prob.n):
            obj = OneVarQuadraticFunction(0, 0, 0)
            nfs = [f.get_onevar_func(x, i) for f in prob.fs]
            nfs = [f for f in nfs if f.P != 0 or f.q != 0]
            viol = max([f.violation(x[i]) for f in nfs])
            logging.debug("Current violation in x[%d]: %.3f", i, viol)
            logging.debug("Current point: %s", x)
            new_xi = x[i]
            new_viol = viol
            ss, es = -tol, viol - viol_tol
            while es - ss > tol:
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
        #if viol_last <= viol + tol:
        #    break
        viol_last = viol

    return x


# phase 2: optimize objective over feasible points
def coord_descent_phase2(x0, prob, num_iters=1000,
    viol_tol=1e-2, tol=1e-4):
    # TODO: find correct termination condition with tolerance
    logging.info("Phase 2 starts")
    x = np.copy(x0)
    viol = max(prob.violations(x))
    update_counter = 0
    converged = False
    for t in range(num_iters):
        # optimize over x[i]
        for i in range(prob.n):
            obj = prob.f0.get_onevar_func(x, i)
            nfs = [f.get_onevar_func(x, i) for f in prob.fs]
            # TODO: maybe this shouldn't be here?
            nfs = [f for f in nfs if f.P != 0 or f.q != 0]
            new_xi = onevar_qcqp(obj, nfs, viol)
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


def improve_coord_descent(x, prob, *args, **kwargs):
    num_iters = kwargs.get('num_iters', 1000)
    viol_tol = kwargs.get('viol_tol', 1e-2)
    tol = kwargs.get('tol', 1e-4)
    phase1 = kwargs.get('phase1', True)

    if phase1:
        x = coord_descent_phase1(x, prob, num_iters, viol_tol, tol)
    if max(prob.violations(x)) < viol_tol:
        x = coord_descent_phase2(x, prob, num_iters, viol_tol, tol)

    return x


def admm_phase1(x0, prob, tol=1e-2, num_iters=1000):
    logging.info("Starting ADMM phase 1")

    z = np.copy(x0)
    xs = [np.copy(x0) for i in range(prob.m)]
    us = [np.zeros(prob.n) for i in range(prob.m)]

    for t in range(num_iters):
        if max(prob.violations(z)) < tol:
            break
        z = (sum(xs)-sum(us))/prob.m
        for i in range(prob.m):
            x, u, f = xs[i], us[i], prob.fi(i)
            xs[i] = onecons_qcqp(z + u, f)
        for i in range(prob.m):
            us[i] += z - xs[i]

    return z


def admm_phase2(x0, prob, rho, tol=1e-2, num_iters=1000, viol_lim=1e4):
    logging.info("Starting ADMM phase 2 with rho %.3f", rho)

    bestx = np.copy(x0)

    z = np.copy(x0)
    xs = [np.copy(x0) for i in range(prob.m)]
    us = [np.zeros(prob.n) for i in range(prob.m)]

    if prob.rho != rho:
        prob.rho = rho
        zlhs = 2*(prob.f0.P + rho*prob.m*sp.identity(prob.n))
        prob.z_solver = SLA.factorized(zlhs)

    last_z = None
    for t in range(num_iters):
        rhs = 2*rho*(sum(xs)-sum(us)) - prob.f0.qarray
        z = prob.z_solver(rhs)

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

        if maxviol > viol_lim: break
        bestx = np.copy(prob.better(z, bestx))

    return bestx


def improve_admm(x0, prob, *args, **kwargs):
    num_iters = kwargs.get('num_iters', 1000)
    viol_lim = kwargs.get('viol_lim', 1e4)
    tol = kwargs.get('tol', 1e-2)
    rho = kwargs.get('rho', None)
    phase1 = kwargs.get('phase1', True)

    if rho is not None:
        lmb0, P0Q = map(np.asmatrix, LA.eigh(prob.f0.P.todense()))
        lmb_min = np.min(lmb0)
        if lmb_min + prob.m*rho < 0:
            logging.error("rho parameter is too small, z-update not convex.")
            logging.error("Minimum possible value of rho: %.3f\n", -lmb_min/prob.m)
            logging.error("Given value of rho: %.3f\n", rho)
            raise Exception("rho parameter is too small, need at least %.3f." % rho)

    # TODO: find a reasonable auto parameter
    if rho is None:
        lmb0, P0Q = map(np.asmatrix, LA.eigh(prob.f0.P.todense()))
        lmb_min = np.min(lmb0)
        lmb_max = np.max(lmb0)
        if lmb_min < 0: rho = 2.*(1.-lmb_min)/prob.m
        else: rho = 1./prob.m
        rho *= 50.
        logging.warning("Automatically setting rho to %.3f", rho)

    if phase1:
        x1 = prob.better(x0, admm_phase1(x0, prob, tol, num_iters))
    else:
        x1 = x0
    x2 = prob.better(x1, admm_phase2(x1, prob, rho, tol, num_iters, viol_lim))
    return x2


def improve_dccp(x0, prob, *args, **kwargs):
    try:
        import dccp
    except ImportError:
        raise Exception("DCCP package is not installed.")

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


def improve_ipopt(x0, prob, *args, **kwargs):
    try:
        import pyipopt
    except ImportError:
        raise Exception("PyIpopt package is not installed.")

    lb = pyipopt.NLP_LOWER_BOUND_INF
    ub = pyipopt.NLP_UPPER_BOUND_INF
    g_L = np.zeros(prob.m)
    for i in range(prob.m):
        if prob.fs[i].relop == '<=':
            g_L[i] = lb
    g_U = np.zeros(prob.m)

    def eval_grad_f(x, user_data = None):
        return 2*prob.f0.P.dot(x) + prob.f0.qarray
    def eval_g(x, user_data = None):
        return np.array([f.eval(x) for f in prob.fs])

    jac_grid = np.indices((prob.m, prob.n))
    jac_r = jac_grid[0].ravel()
    jac_c = jac_grid[1].ravel()
    def eval_jac_g(x, flag, user_data = None):
        if flag:
            return (jac_r, jac_c)
        else:
            return np.vstack([2*f.P.dot(x)+f.qarray for f in prob.fs])

    nlp = pyipopt.create(
        prob.n, lb*np.ones(prob.n), ub*np.ones(prob.n),
        prob.m, g_L, g_U, prob.m*prob.n, 0,
        prob.f0.eval, eval_grad_f,
        eval_g, eval_jac_g
    )
    try:
        x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
    except:
        pass

    return x


class QCQP:
    def __init__(self, prob):
        self.prob = prob
        self.qcqp_form = get_qcqp_form(prob)
        self.n = self.qcqp_form.n
        self.spectral_sol = None
        self.spectral_bound = None
        self.sdr_sol = None
        self.sdr_bound = None
        self.maximize_flag = (prob.objective.NAME == "maximize")

    def suggest(self, method=s.RANDOM, eps=1e-8, *args, **kwargs):
        if method not in s.suggest_methods:
            raise Exception("Unknown suggest method: %s\n", method)
        if method == s.RANDOM:
            x = np.random.randn(self.n)
        elif method == s.SPECTRAL:
            if self.spectral_sol is None:
                self.spectral_sol, self.spectral_bound = solve_spectral(self.qcqp_form, *args, **kwargs)
                if self.maximize_flag:
                    self.spectral_bound *= -1
            x = self.spectral_sol
        elif method == s.SDR:
            if self.sdr_sol is None:
                self.sdr_sol, self.sdr_bound = solve_sdr(self.qcqp_form, *args, **kwargs)
                if self.maximize_flag:
                    self.sdr_bound *= -1
                self.mu = np.asarray(self.sdr_sol[:-1, -1]).flatten()
                self.Sigma = self.sdr_sol[:-1, :-1] - self.mu*self.mu.T + eps*sp.identity(self.n)
            x = np.random.multivariate_normal(self.mu, self.Sigma)

        assign_vars(self.prob.variables(), x)
        f0 = self.qcqp_form.f0.eval(x)
        if self.maximize_flag: f0 *= -1
        return (f0, max(self.qcqp_form.violations(x)))

    def _improve(self, method, *args, **kwargs):
        x0 = flatten_vars(self.prob.variables(), self.n)
        if method == s.COORD_DESCENT:
            x = improve_coord_descent(x0, self.qcqp_form, args, kwargs)
        elif method == s.ADMM:
            x = improve_admm(x0, self.qcqp_form, args, kwargs)
        elif method == s.DCCP:
            x = improve_dccp(x0, self.qcqp_form, args, kwargs)
        elif method == s.IPOPT:
            x = improve_ipopt(x0, self.qcqp_form, args, kwargs)

        assign_vars(self.prob.variables(), x)
        f0 = self.qcqp_form.f0.eval(x)
        if self.maximize_flag: f0 *= -1
        return (f0, max(self.qcqp_form.violations(x)))


    def improve(self, method, *args, **kwargs):
        if not isinstance(method, list): methods = [method]
        else: methods = method

        if not all([method in s.improve_methods for method in methods]):
            raise Exception("Unknown improve method(s): ", methods)

        if any([x is None for x in self.prob.variables()]):
            self.suggest()

        for method in methods:
            f, v = self._improve(method, args, kwargs)
        return (f, v)
