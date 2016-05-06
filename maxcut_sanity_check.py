import sys
sys.path.append('/home/jaehyun/qcqp/qcqp')
import cvxpy as cvx
import qcqp
import numpy as np

n = 50
m = 200
s = 290797
W = np.zeros((n, n))
for k in range(m):
    i = s%n
    s = (s*s)%50515093
    j = s%n
    s = (s*s)%50515093
    W[i, j] = W[j, i] = 1

# Find upper bound using SDP relaxation
x = cvx.Variable(n)
obj = 0.25*(cvx.sum_entries(W) - qcqp.quad_form(x, W))
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