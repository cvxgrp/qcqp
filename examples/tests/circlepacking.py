import cvxpy as cvx
import numpy as np
import cvxopt
import matplotlib.pyplot as plt
import qcqp
import dccp

N = 3 # number of circles
X = cvx.Variable(2, N)
B = 10
r = cvx.Variable()
obj = cvx.Maximize(r)
cons = [X >= r, X <= B-r]
for i in range(N):
    for j in range(i+1, N):
        cons.append(cvx.sum_squares(X[:, i]-X[:, j]) - 4*cvx.square(r) >= 0)

prob = cvx.Problem(obj, cons)
# lb = prob.solve(method="sdp-relax",solver=cvx.MOSEK)
# print(lb)
print(prob.solve(method="qcqp-admm",num_samples=5,num_iters=20))
#prob.solve(method="dccp") # not a dccp problem
#print(prob.solve(method="qcqp-dccp",num_samples=10))

# plot the circles
circ = np.linspace(0,2 * np.pi)
for i in xrange(N):
    plt.plot(X[0, i].value+r.value*np.cos(circ),X[1, i].value+r.value*np.sin(circ),'b')
plt.xlim([0, B])
plt.ylim([0, B])
plt.axes().set_aspect('equal')
plt.show()
