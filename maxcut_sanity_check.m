n = 20;
m = 50;
s = 290797;
W = zeros(n, n);
for k=1:m
    i = mod(s, n); s = mod(s*s, 50515093);
    j = mod(s, n); s = mod(s*s, 50515093);
    W(i+1, j+1) = 1;
    W(j+1, i+1) = 1;
end

cvx_begin quiet
    variables X(n, n) x(n)
    maximize 0.25 * ( sum_entries(W) - trace(W*X) )
    subject to
        diag(X) == 1
        [X, x; x', 1] == semidefinite(n+1)
cvx_end

ub = cvx_optval
xr = round(x);
lb = 0.25*(sum(sum(W)) - xr'*W*xr)