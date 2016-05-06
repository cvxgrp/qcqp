import cvxpy as cvx
import numpy as np

np.random.seed(1)
n = 2

#X = cvx.Variable(n, n)
X = cvx.Symmetric(n)
W = np.random.randn(n, n)

obj = cvx.sum_entries(cvx.mul_elemwise(W, X))
cons = [X == 1, X >> 0]
#cons = [X == 1]
prob = cvx.Problem(cvx.Minimize(obj), cons)

print (prob.solve(solver=cvx.MOSEK))
