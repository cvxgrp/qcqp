# TODO: remove this and make CVXPY quad_form return
#   sum_squares(L1*x) - sum_squares(L2*x)   (difference-of-convex form)
def quad_form(x, P):
    return x.T * P * x
