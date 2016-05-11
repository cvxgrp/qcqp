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
import canonInterface
import cvxpy.lin_ops.lin_utils as lu

# Take an arbitrary indefinite quadratic expression and return an array
# of matrices, where each entry is a sparse matrix containing the
# coefficients of the corresponding entry in the original expression
# The array is flattened in the column first order
def lift(expr, id_to_col, N):
    if expr.is_constant():
        ret = [sp.lil_matrix((N+1, N+1)) for i in range(expr.size[0]*expr.size[1])]
        row = 0
        for j in range(expr.size[1]):
            for i in range(expr.size[0]):
                ret[row][N, N] = expr.value[i, j]
                row += 1
        return ret
    elif expr.is_affine():
        s, _ = expr.canonical_form
        V, I, J, b = canonInterface.get_problem_matrix(
            [lu.create_eq(s)], id_to_col, None)
        shape = (N, expr.size[0]*expr.size[1])
        A = sp.coo_matrix((V, (J, I)), shape=shape).tocsc()
        ret = [sp.lil_matrix((N+1, N+1)) for i in range(expr.size[0]*expr.size[1])]
        row = 0
        for j in range(expr.size[1]):
            for i in range(expr.size[0]):
                ret[row][0:N, N] = A[:, row]
                ret[row][N, N] = b[row]
                row += 1
        return ret
    
    assert expr.is_quadratic()
    

    if isinstance(expr, cvx.affine_prod):
        Xs = lift(expr.args[0], id_to_col, N)
        Ys = lift(expr.args[1], id_to_col, N)
        pass
    elif isinstance(expr, cvx.quad_over_lin):
        Xs = lift(expr.args[0], id_to_col, N)
        y = expr.args[1].value
        pass
    elif isinstance(expr, cvx.power):
        Xs = lift(expr.args[0], id_to_col, N)
        p = expr.args[1].value
        pass
    elif isinstance(expr, cvx.matrix_frac):
        Xs = lift(expr.args[0], id_to_col, N)
        P = expr.args[1].value
        pass
    elif isinstance(expr, cvx.affine.affine_atom.AffAtom):
        Xs = [X for arg in expr.args for X in lift(arg, id_to_col, N)]
        fake_args = []
        var_offsets = {}
        offset = 0
        for idx, arg in enumerate(expr.args):
            if arg.is_constant():
                fake_args += [lu.create_const(arg.value, arg.size)]
            else:
                fake_args += [lu.create_var(arg.size, idx)]
                var_offsets[idx] = offset
                offset += arg.size[0]*arg.size[1]
        fake_expr, _ = expr.graph_implementation(fake_args, expr.size,
                                                 expr.get_data())
        # Get the matrix representation of the function.
        V, I, J, b = canonInterface.get_problem_matrix(
            [lu.create_eq(fake_expr)], var_offsets, None)
        # return "AX+b"
        ret = [sp.csc_matrix((N+1, N+1)) for i in range(expr.size[0]*expr.size[1])]
        for (v, i, j) in zip(V, I, J):
            ret[i] += v*Xs[j]
        for i, v in enumerate(b):
            ret[i][N, N] += v
        return ret
    else:
        raise Exception("Unknown expression type %s." % type(expr))


# Take a scalar-valued indefinite quadratic expression and lift it to the form of
#   Tr(MX),
# where M = [P, q; q^T, r], X = [Z, x; x^T, 1] of appropriate dimensions
def lift2(expr, X, id_to_col, N):
    for var in expr.variables():
        var.value = np.zeros(var.size)
    r = expr.value

    q = np.asmatrix(np.zeros((N, 1)))
    g = expr.grad
    
    for var in expr.variables():
        s_ind = id_to_col[var.id]
        e_ind = s_ind + var.size[0]*var.size[1]
        q[s_ind:e_ind] = g[var].todense()

    P = np.asmatrix(np.zeros((N, N)))

    for var in expr.variables():
        col_ind = id_to_col[var.id]
        for j in range(var.size[1]):
            for i in range(var.size[0]):
                var.value[i, j] = 0.5
                g = expr.grad

                for var2 in expr.variables():
                    s_ind = id_to_col[var2.id]
                    e_ind = s_ind + var2.size[0]*var2.size[1]
                    P[s_ind:e_ind, col_ind] = g[var2].todense()

                P[:, col_ind] -= q
                
                var.value[i, j] = 0
                col_ind += 1

    M = np.bmat([[P, q], [np.asmatrix(np.zeros((1, N))), np.asmatrix(r)]])
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
    id_to_col, N = get_id_map(self.variables())

    # lifted variables and semidefinite constraint
    X = cvx.Semidef(N + 1)
    
    rel_obj = type(self.objective)(lift2(self.objective.args[0], X, id_to_col, N))
    # M = lift(self.objective.args[0], id_to_col, N)[0]
    # rel_obj = type(self.objective)(M*X)
    rel_constr = [X[N, N] == 1]
    
    # for constr in self.constraints:
    #     Ms = lift(constr._expr, id_to_col, N)
    #     for M in Ms:
    #         if constr.OP_NAME == '==':
    #             rel_constr.append(M*X == 0)
    #         else:
    #             rel_constr.append(M*X <= 0)
    for constr in self.constraints:
        for i in range(constr._expr.size[0]):
            for j in range(constr._expr.size[1]):
                c = lift2(constr._expr[i, j], X, id_to_col, N)
                if constr.OP_NAME == '==':
                    rel_constr.append(c == 0)
                else:
                    rel_constr.append(c <= 0)

    rel_prob = cvx.Problem(rel_obj, rel_constr)
    rel_prob.solve(*args, **kwargs)
    
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
