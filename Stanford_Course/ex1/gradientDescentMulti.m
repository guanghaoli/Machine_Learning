function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
%     temp1 = theta(1) - alpha*(sum((X*theta-y).*X(:,1)))/m;
%     temp2 = theta(2) - alpha*(sum((X*theta-y).*X(:,2)))/m;
%     temp3 = theta(3) - alpha*(sum((X*theta-y).*X(:,3)))/m;
%     theta(1) = temp1;
%     theta(2) = temp2;
%     theta(3) = temp3;
   temp = [];
   len = size(X,2);
    for i=1:len
        temp(i) = theta(i) - alpha*(sum((X*theta-y).*X(:,i)))/m;
    end
    for j=1:len
        theta(j) = temp(j);    
    end
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
