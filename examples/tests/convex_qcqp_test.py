from cvxpy import *
import numpy
import qcqp

# Problem data.
m = 5
n = 3
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m)

# Construct the problem.
x = Variable(n)
objective = Minimize(sum_squares(A*x - b))
constraints = [0 <= x, x <= 1]
prob = Problem(objective, constraints)

print(prob.solve())
print x.value

# The optimal objective is returned by prob.solve().
result = prob.solve(method='qcqp-admm', num_samples=5, num_iters=100)
# The optimal value for x is stored in x.value.
print x.value
