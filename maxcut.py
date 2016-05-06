import sys
sys.path.append('/home/jaehyun/qcqp/qcqp')
import cvxpy as cvx
import qcqp
import numpy as np

# Maximum cut problem.
np.random.seed(1)
n = 20

# Make each edge with probability p
p = 0.2

# Make adjacency matrix.
W = np.asmatrix(np.zeros( (n, n) ))
for i in range(n):
    for j in range(i+1, n):
        if np.random.uniform() < p:
            W[i, j] = W[j, i] = 1

# Find upper bound using SDP relaxation
x = cvx.Variable(n)
obj = 0.25*(cvx.sum_entries(W) - x.T*W*x)
cons = [cvx.square(x) == 1]
prob = cvx.Problem(cvx.Maximize(obj), cons)

ub = prob.solve(method='relax-SDP', solver=cvx.MOSEK)
print ('Upper bound: %.3f' % ub)

x_round = np.sign(x.value)
lb = 0.25*(np.sum(W) - (x_round.T*W*x_round)[0, 0])
print ('Lower bound: %.0f' % lb)


#print (W)
#print (x.value)
#print (x_round)