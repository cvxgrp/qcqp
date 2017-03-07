#!/usr/bin/python
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import qcqp

N = 5 # number of circles
X = cvx.Variable(2, N)
B = 10
r = cvx.Variable()
obj = cvx.Maximize(r)
cons = [X >= r, X <= B-r, r >= 0]
for i in range(N):
    for j in range(i+1, N):
        cons.append(cvx.square(2*r) <= cvx.sum_squares(X[:, i]-X[:, j]))

prob = cvx.Problem(obj, cons)

# ub, lb_cd, lb_admm, lb_dccp = 0, 0, 0, 0
ub = prob.solve(method='sdp-relax', solver=cvx.MOSEK)
lb_cd = prob.solve(method='coord-descent', solver=cvx.MOSEK, num_samples=1)
lb_admm = prob.solve(method='qcqp-admm', solver=cvx.MOSEK, num_samples=10, num_iters=10, rho=10)
lb_dccp = prob.solve(method='qcqp-dccp', solver=cvx.MOSEK, num_samples=10)

print ('Upper bound: %.3f' % ub)
print ('Lower bounds:')
print ('  Coordinate descent: %.3f' % lb_cd)
print ('  Nonconvex ADMM: %.3f' % lb_admm)
print ('  Convex-concave programming: %.3f' % lb_dccp)

print('r:       %.3f' % r.value)
print('Point 1: %.3f %.3f' % (X[0, 0].value, X[1, 0].value))
print('Point 2: %.3f %.3f' % (X[0, 1].value, X[1, 1].value))

# plot the circles
circ = np.linspace(0,2 * np.pi)
for i in xrange(N):
    plt.plot(
        X[0, i].value+r.value*np.cos(circ),
        X[1, i].value+r.value*np.sin(circ), 'b'
    )
plt.xlim([0, B])
plt.ylim([0, B])
plt.axes().set_aspect('equal')
plt.show()
