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
y_Vec = (1:num_labels) == y;    #Create a 1x10 vector for each output value, y, such that the vector is "0" for 9 and "1" at the location given by the value of y
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

for item = 1: m
  for label=1:num_labels
    z2 = [1 X(item,:)]*Theta1';       % Size = 1 401 times 401 25 
    a2 = sigmoid(z2);                 % Size = 1 25
    a2 = [ones(size(a2,1),1) a2];     % Size = 1 26
    z3 = a2*Theta2';                  % Size = 1 26 times 26 10
    a3 = sigmoid(z3);                 % Size = 1 10
    first = (log(a3(label))*y_Vec(item,label)) ;                  % Number
    second = ((1- y_Vec(item,label))'*log(1 - a3(label))');       % Number 
    J += (1/m)*(-first-second);                                   % Number 
  end 
end
%Add the regularization term to the computed cost 
%Remember not to regularize the bias term, the first column in the Theta1 and Theta2 matrix 
regu = (lambda/(2*m))*(sum(sum(Theta1 .^ 2)) - sum(Theta1(:,1) .^ 2)) + (lambda/(2*m))*(sum(sum(Theta2 .^ 2)) - sum(Theta2(:,1) .^ 2));
J += regu;


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

  
% Add ones to all the training sets 
A1 = [ones(m,1) X];  % size = 5000 x 401
Z2 = A1 * Theta1';   % size = 5000 x 25
A2 = sigmoid(Z2);    % size = 5000 x 25

% Add 1's for the bias unit, size = 5000 x 26
A2 = [ones(size(A2,1),1), A2];
Z3 = A2 * Theta2';   % Size = 5000 x 10
A3 = sigmoid(Z3);    % Size = 5000 x 10
  

d3 = A3 - y_Vec; % Size = 5000 x 10
d2 = (d3 * Theta2) .* [ones(size(Z2,1),1) sigmoidGradient(Z2)]; % Size = 5000 x 26
% Need to remove d2 calculated for the bias node, first column of the d2
d2 = d2(:,2:end);     % Size = 5000 x 25 

% Compute the del matrices
del1 = d2' * A1;
del2 = d3' * A2;
 
% Compute gradients from the del matrices
Theta1_grad = (1/m) * del1; % 25 x 401
Theta2_grad = (1/m) * del2; % 10 x 26

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%VVI: Do not regularize the gradient matrices first column, always coming from the bias 
Theta1_grad += (1.0/m)*[zeros(size(Theta1,1),1) lambda*Theta1(:,2:end)];
Theta2_grad += (1.0/m)*[zeros(size(Theta2,1),1) lambda*Theta2(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
