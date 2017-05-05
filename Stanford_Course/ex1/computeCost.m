function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% u1 = mean(X);
% M1 = max(X);
% m1 = min(X);
% u2 = mean(y);
% M2 = max(y);
% m2 = min(y);
% X(:,2) = (X(:,2)-u1(2))/(M1(2)-m1(2));
% y = (y-u2)/(M2-m2);

J = sum((X*theta-y).^2)/(2*m);
% for i=1:m
%     J = J + (X(1,:)*theta-y(i))^2;    
% end
% J = J/(2*m);



% =========================================================================

end
