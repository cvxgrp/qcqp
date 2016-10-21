import numpy
import cvxpy as cvx
import qcqp

n = 3
numpy.random.seed(1)

# Make adjacency matrix.
W = numpy.random.binomial(1, 0.2, size=(n, n))
W = numpy.asmatrix(W)

# Solve using SDP relaxation
x = cvx.Variable(n)
obj = 0.25*(cvx.sum_entries(W) - cvx.quad_form(x, W))
cons = [cvx.square(x) == 1]
prob = cvx.Problem(cvx.Maximize(obj), cons)

ub = prob.solve(method='sdp-relax', solver=cvx.MOSEK)
print ('Upper bound: %.3f' % ub)

prob.solve(method='qcqp-admm',num_samples=5,num_iters=3)
#print ('Upper bound: %.3f' % ub)
