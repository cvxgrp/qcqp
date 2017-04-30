#!/usr/bin/python
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
from qcqp import *

n = 5 # number of circles
X = cvx.Variable(2, n)
B = 10
r = cvx.Variable()
obj = cvx.Maximize(r)
cons = [X >= r, X <= B-r, r >= 0]
for i in range(n):
    for j in range(i+1, n):
        cons.append(cvx.square(2*r) <= cvx.sum_squares(X[:, i]-X[:, j]))

prob = cvx.Problem(obj, cons)
qcqp = QCQP(prob)

# sample from the semidefinite relaxation
qcqp.suggest(SDR)
print("SDR-based upper bound: %.3f" % qcqp.sdr_bound)

f_cd, v_cd = qcqp.improve(COORD_DESCENT)
print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))

# SDR solution is cached and not solved again
qcqp.suggest(SDR)
f_dccp, v_dccp = qcqp.improve(DCCP)
print("Penalty CCP: objective %.3f, violation %.3f" % (f_dccp, v_dccp))
X_dccp = np.copy(X.value)
r_dccp = r.value

qcqp.suggest(SDR)
f_admm, v_admm = qcqp.improve(ADMM)
print("Nonconvex ADMM: objective %.3f, violation %.3f" % (f_admm, v_admm))

# plot the circles
circ = np.linspace(0, 2*np.pi)
for i in xrange(n):
    plt.plot(
        X_dccp[0, i]+r_dccp*np.cos(circ),
        X_dccp[1, i]+r_dccp*np.sin(circ), 'b'
    )
plt.xlim([0, B])
plt.ylim([0, B])
plt.axes().set_aspect('equal')
plt.show()
