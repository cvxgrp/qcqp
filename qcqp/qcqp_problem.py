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
import scipy.sparse as sp
import cvxpy.lin_ops.lin_utils as lu
from numpy import linalg as LA
from cvxpy.utilities import QuadCoeffExtractor

def get_id_map(vars):
    id_map = {}
    N = 0
    for x in vars:
        id_map[x.id] = N
        N += x.size[0]*x.size[1]
    return id_map, N

def solve_SDP_relaxation(prob, *args, **kwargs):
    """Solve the SDP relaxation.
    """
    # check quadraticity
    if not prob.objective.args[0].is_quadratic():
        raise Exception("Objective is not quadratic.")
    if not all([constr._expr.is_quadratic() for constr in prob.constraints]):
        raise Exception("Not all constraints are quadratic.")
    if prob.is_dcp():
        warnings.warn("Problem is already convex; relaxation is not necessary.")
    warnings.warn("Solving SDP relaxation of nonconvex QCQP.")
    
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

def relax_sdp(self, *args, **kwargs):
    X, sdp_bound, id_map, N = solve_SDP_relaxation(self, *args, **kwargs)
    
    ind = 0
    for x in self.variables():
        size = x.size[0]*x.size[1]
        x.value = np.reshape(X[ind:ind+size, -1], x.size, order='F')
        ind += size

    return sdp_bound

def noncvx_admm(self, *args, **kwargs):
    X, sdp_bound, id_map, N = solve_SDP_relaxation(self, *args, **kwargs)

    num_samples = 100
    num_iters = 100
    eps = 0.001

    # TODO
    # (1) generate random samples using X
    # (2) run ADMM on each sample

    return NotImplemented

def qcqp_dccp(self, *args, **kwargs):
    # TODO: don't solve SDP relaxation, split the given problem
    # into convex/concave parts
    return NotImplemented

# Add solution methods to Problem class.
cvx.Problem.register_solve("relax-SDP", relax_sdp)
cvx.Problem.register_solve("noncvx-admm", noncvx_admm)
cvx.Problem.register_solve("qcqp-dccp", qcqp_dccp)
