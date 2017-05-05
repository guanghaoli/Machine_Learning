function theta = normal(X, y)
theta = pinv(X'*X)*X'*y;
end
