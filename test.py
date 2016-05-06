import sys
sys.path.append('/home/jaehyun/qcqp/qcqp')
from cvxpy import *
import qcqp
import numpy as np

np.random.seed(1)
n = 3

W = np.random.randn(n, n)

x = Variable(n)
y = Variable(n)
z = Variable(n)
obj = qcqp.quad_form(x, W) + sum_entries(y.T*z) - sum_squares(x-y)
cons = [square(x) == 1, -1 <= y, y <= 2, sum_squares(z) >= 5]
prob = Problem(Maximize(obj), cons)

ub = prob.solve(method='relax-SDP', solver=MOSEK)
print ('Upper bound: %.3f' % ub)
