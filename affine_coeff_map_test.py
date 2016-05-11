from __future__ import division
import warnings
import cvxpy as cvx
import numpy as np
import scipy.sparse as sp
import time
import canonInterface
import sys
import abc
import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.atom import Atom
import operator as op
import canonInterface
import scipy.sparse as sp

def get_id_map(vars):
    id_to_col = {}
    N = 0
    for x in vars:
        id_to_col[x.id] = N
        N += x.size[0]*x.size[1]
    return id_to_col, N

AA=np.random.randn(3, 2)
BB=np.random.randn(3, 3)
cc=np.random.randn(3, 2)
x=cvx.Variable(2, 2)
y=cvx.Variable(3, 2)
expr=(AA*x+BB*y+cc)

id_to_col, N = get_id_map(expr.variables())
s, _ = expr.canonical_form
V, I, J, b = canonInterface.get_problem_matrix([lu.create_eq(s)], id_to_col, None)
shape = (N, expr.size[0]*expr.size[1])
A = sp.coo_matrix((V, (J, I)), shape=shape).tocsc()
