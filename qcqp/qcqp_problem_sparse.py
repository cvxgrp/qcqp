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
import time
import canonInterface

# Take a scalar-valued indefinite quadratic expression and lift it to the form of
#   Tr(MX),
# where M = [P, q; q^T, r], X = [Z, x; x^T, 1] of appropriate dimensions
def lift(expr, X, id_to_col, N):
    for var in expr.variables():
        var.value = np.zeros(var.size)
    r = expr.value

    q = sp.lil_matrix((N, 1))
    g = expr.grad
    
    for var in expr.variables():
        s_ind = id_to_col[var.id]
        e_ind = s_ind + var.size[0]*var.size[1]
        q[s_ind:e_ind] = g[var]

    datas = []
    rows = []
    cols = []
    for var in expr.variables():
        col_ind = id_to_col[var.id]
        for j in range(var.size[1]):
            for i in range(var.size[0]):
                var.value[i, j] = 0.5
                g = expr.grad

                for var2 in expr.variables():
                    data = g[var2].data
                    #print (g[var2].indices)
                    #print (id_to_col[var2.id])
                    #print (g[var2].indices + id_to_col[var2.id])
                    row = g[var2].indices + id_to_col[var2.id]
                    col = np.full(row.shape, col_ind, dtype=np.int32)
                    rows.append(row)
                    cols.append(col)
                    datas.append(data)

                col_ind += 1

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    data = np.concatenate(datas)
    P = sp.coo_matrix((data, (row, col)), shape=(N, N)).tolil()

    for var in expr.variables():
        col_ind = id_to_col[var.id]
        for j in range(var.size[1]):
            for i in range(var.size[0]):
                P[:, col_ind] -= q
                col_ind += 1

    M = sp.bmat([[P, q], [None, sp.coo_matrix([[r]])]])
    return cvx.sum_entries(cvx.mul_elemwise(M, X))

def get_id_map(vars):
    id_to_col = {}
    N = 0
    for x in vars:
        id_to_col[x.id] = N
        N += x.size[0]*x.size[1]
    return id_to_col, N

def relax_sdp(self, *args, **kwargs):
    """Solve the SDP relaxation.
    """
    # check quadraticity
    if not self.objective.args[0].is_quadratic():
        raise Exception("Objective function is not quadratic.")
    if not all([constr._expr.is_quadratic() for constr in self.constraints]):
        raise Exception("Not all constraints are quadratic.")

    if self.is_dcp():
        # TODO: redirect this to normal solve method?
        warnings.warn("Problem is DCP; SDP relaxation is unnecessary.")

    warnings.warn("Solving an SDP relaxation of a nonconvex QCQP.")
    t1 = time.time()
    id_to_col, N = get_id_map(self.variables())

    # lifted variables and semidefinite constraint
    X = cvx.Semidef(N + 1)
    
    rel_obj = type(self.objective)(lift(self.objective.args[0], X, id_to_col, N))
    rel_constr = [X[N, N] == 1]
    for constr in self.constraints:
        for i in range(constr._expr.size[0]):
            for j in range(constr._expr.size[1]):
                c = lift(constr._expr[i, j], X, id_to_col, N)
                if constr.OP_NAME == '==':
                    rel_constr.append(c == 0)
                else:
                    rel_constr.append(c <= 0)

    rel_prob = cvx.Problem(rel_obj, rel_constr)
    t2 = time.time()
    rel_prob.solve(*args, **kwargs)
    t3 = time.time()
    print "%.2f secs for problem conversion" % (t2-t1)
    print "%.2f secs for problem solving" % (t3-t2)
    
    if rel_prob.status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
        print (rel_prob.status)
        return rel_prob.value

    ind = 0
    for x in self.variables():
        for i in range(x.size[0]):
            for j in range(x.size[1]):
                x.value[i, j] = X.value[ind, -1]
                ind += 1

    return rel_prob.value

# Add SDP relaxation method to cvx Problem.
cvx.Problem.register_solve("relax-SDP", relax_sdp)
