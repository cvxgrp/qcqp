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
import canonInterface
import cvxpy.lin_ops.lin_utils as lu
from numpy import linalg as LA

def quad_coeffs_constant(expr, id_map, N):
    ret = [sp.lil_matrix((N+1, N+1)) for i in range(expr.size[0]*expr.size[1])]
    if expr.is_scalar():
        ret[0][N, N] = expr.value
    else:
        row = 0
        for j in range(expr.size[1]):
            for i in range(expr.size[0]):
                ret[row][N, N] = expr.value[i, j]
                row += 1
    return ret

def quad_coeffs_affine(expr, id_map, N):
    s, _ = expr.canonical_form
    V, I, J, b = canonInterface.get_problem_matrix([lu.create_eq(s)], id_map)
    ret = [sp.lil_matrix((N+1, N+1)) for i in range(expr.size[0]*expr.size[1])]
    for (v, i, j) in zip(V, I, J):
        ret[int(i)][j, N] = v
    for i, v in enumerate(b):
        ret[i][N, N] += v
    return ret

def quad_coeffs_affine_prod(expr, id_map, N):
    Xs = quad_coeffs(expr.args[0], id_map, N)
    Ys = quad_coeffs(expr.args[1], id_map, N)
    ret = []
    m, p = expr.args[0].size
    n = expr.args[1].size[1]
    for j in range(n):
        for i in range(m):
            M = sp.lil_matrix((N+1, N+1))
            for k in range(p):
                Xind = k*m + i
                Yind = j*p + k
                M += Xs[Xind] * Ys[Yind].T
            ret.append(M)
    return ret

def quad_coeffs_quad_over_lin(expr, id_map, N):
    Xs = quad_coeffs(expr.args[0], id_map, N)
    y = expr.args[1].value
    return [sum([X*X.T for X in Xs]) / y]

def quad_coeffs_power(expr, id_map, N):
    Xs = quad_coeffs(expr.args[0], id_map, N)
    if expr.p == 1:
        return Xs
    elif expr.p == 2:
        return [X*X.T for X in Xs]
    else:
        raise Exception("Error while processing power(x, %f)." % p)

def quad_coeffs_matrix_frac(expr, id_map, N):
    Xs = quad_coeffs(expr.args[0], id_map, N)
    Pinv = LA.inv(expr.args[1].value)
    m, n = expr.args[0].size
    M = sp.lil_matrix((N+1, N+1))
    for i in range(m):
        for j in range(m):
            M += sum([Pinv[i, j]*Xs[i+k*m]*Xs[j+k*m].T for k in range(n)])
    return [M]

def quad_coeffs_affine_atom(expr, id_map, N):
    Xs = []
    fake_args = []
    offsets = {}
    offset = 0
    for idx, arg in enumerate(expr.args):
        if arg.is_constant():
            fake_args += [lu.create_const(arg.value, arg.size)]
        else:
            Xs += quad_coeffs(arg, id_map, N)
            fake_args += [lu.create_var(arg.size, idx)]
            offsets[idx] = offset
            offset += arg.size[0]*arg.size[1]
    fake_expr, _ = expr.graph_implementation(
        fake_args, expr.size, expr.get_data())
    # Get the matrix representation of the function.
    V, I, J, b = canonInterface.get_problem_matrix([lu.create_eq(fake_expr)], offsets)
    # return "AX+b"
    ret = [sp.lil_matrix((N+1, N+1)) for i in range(expr.size[0]*expr.size[1])]
    for (v, i, j) in zip(V, I, J):
        ret[int(i)] += v*Xs[int(j)]
    for i, v in enumerate(b):
        ret[i][N, N] += v
    return ret

# Take an arbitrary indefinite quadratic expression of size m*n,
# and return an array of length mn, where the (jm+i)th entry is
# the coefficient matrix of the (i, j) entry of the expression.
# The coefficient matrices are of the form [P, u; v^T r], such that
# the coressponding expression is
#   x^T P x + (u+v)^T x + r,
# where x is the flattened array of variables.
def quad_coeffs(expr, id_map, N):
    if expr.is_constant():
        return quad_coeffs_constant(expr, id_map, N)
    elif expr.is_affine():
        return quad_coeffs_affine(expr, id_map, N)
    elif isinstance(expr, cvx.affine_prod):
        return quad_coeffs_affine_prod(expr, id_map, N)
    elif isinstance(expr, cvx.quad_over_lin):
        return quad_coeffs_quad_over_lin(expr, id_map, N)
    elif isinstance(expr, cvx.power):
        return quad_coeffs_power(expr, id_map, N)
    elif isinstance(expr, cvx.matrix_frac):
        return quad_coeffs_matrix_frac(expr, id_map, N)
    elif isinstance(expr, cvx.affine.affine_atom.AffAtom):
        return quad_coeffs_affine_atom(expr, id_map, N)
    else:
        raise Exception("Unknown expression type %s." % type(expr))

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
    # lifted variables and semidefinite constraint
    X = cvx.Semidef(N + 1)
    M = quad_coeffs(prob.objective.args[0], id_map, N)[0]
    rel_obj = type(prob.objective)(cvx.sum_entries(cvx.mul_elemwise(M, X)))
    rel_constr = [X[N, N] == 1]
    for constr in prob.constraints:
        Ms = quad_coeffs(constr._expr, id_map, N)
        for M in Ms:
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

    if prob.objective.NAME == "minimize":
        print ("Lower bound: " + str(rel_prob.value))
    else:
        print ("Upper bound: " + str(rel_prob.value))

    return X.value, rel_prob.value, id_map, N

def relax_sdp(self, *args, **kwargs):
    X, sdp_bound, id_map, N = solve_SDP_relaxation(self, *args, **kwargs)
    
    ind = 0
    for x in self.variables():
        size = x.size[0]*x.size[1]
        x.value = np.reshape(X[ind:ind+size, -1], x.size, order='F')
        ind += size

    return sdp_bound

def relax_sdp_rand(self, *args, **kwargs):
    X, sdp_bound, id_map, N = solve_SDP_relaxation(self, *args, **kwargs)

    # TODO: generate random samples using X
    
    ind = 0
    for x in self.variables():
        size = x.size[0]*x.size[1]
        x.value = np.reshape(X[ind:ind+size, -1], x.size, order='F')
        ind += size

    return sdp_bound

# Add SDP relaxation method to cvx Problem.
cvx.Problem.register_solve("relax-SDP", relax_sdp)
cvx.Problem.register_solve("relax-SDP-rand", relax_sdp_rand)
