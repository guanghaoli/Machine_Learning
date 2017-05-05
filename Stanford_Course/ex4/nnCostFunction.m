function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%% calculate the cost function J
% m is the size of X, the number of rows
X = [ones(m,1) X]; %5000*401
z2 = X*Theta1'; %5000*25
a2 = sigmoid(z2); %5000*25
a2 = [ones(m,1) a2]; % 5000*26
z3 = a2*Theta2'; % 5000*10
a3 = sigmoid(z3); % 5000*10
% the output layer get result with num_labels*1 vector
% so we should switch y as a matrix which is m*num_labels,5000*10 in this
% example
for i=1:m
    yi = zeros(num_labels,1);
    yii = ones(num_labels,1); 
    yi(y(i)) = 1;
    yii(y(i)) = 0;
    J =J-log(a3(i,:)*yi)-log(1-a3(i,:))*yii;
end
J = J/m;
%% add the regularization to J
% 
d1 = hidden_layer_size+1;
d2 = hidden_layer_size * (input_layer_size + 1);
d3 = num_labels+1;
d4 = num_labels*(hidden_layer_size+1);
regu = sum(Theta1(d1:d2).^2)+sum(Theta2(d3:d4).^2);
J = J+regu*lambda/(2*m);

%% back propagation
Y = zeros(m,num_labels); % 5000*10
% switch y as a matrix 5000*10, according to the number of y 
% m*num_labels , 5000*10 matrix in this example
for i=1:m
    Y(i,y(i))=1;
end
delta3 = a3-Y;
TheTa2 = Theta2(:,2:end);
delta2 = delta3*TheTa2.*sigmoidGradient(z2);
Theta1_grad = delta2'*X/m;
Theta2_grad = delta3'*a2/m;

%% regularization of grad
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda*Theta1(:,2:end)/m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda*Theta2(:,2:end)/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
