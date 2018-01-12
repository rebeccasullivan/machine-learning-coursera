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

X = [ones(m, 1) X];

temp = zeros(rows(y), num_labels);
for i = 1:rows(y)
	temp(i, y(i)) = 1;
endfor;

y = temp;

% Compute a2
a1 = X;	
z2 = Theta1 * X';
a2 = sigmoid(z2);
	
% Add bias element to a2
a2 = [ones(1, columns(a2)); a2];

% Compute a3
z3 = Theta2 * a2;
a3 = sigmoid(z3)';

[max, pred] = max(a3, [], 2);

h = a3;

% Compute cost function 
J = (1 / m) * ones(1, m) * ((log(h) .* -y) - (log(1 - h) .* (1 - y))) * ones(num_labels, 1);

% Remove first column of Theta1 and Theta2 to avoid regularizing
Theta1_without_1 = Theta1(:,2:end);
Theta2_without_1 = Theta2(:,2:end);

lambda_factor = lambda / (2 * m);

reg_component_1 = sum((Theta1_without_1 .* Theta1_without_1) * ones(columns(Theta1_without_1), 1));
reg_component_2 = sum((Theta2_without_1 .* Theta2_without_1) * ones(columns(Theta2_without_1), 1));

J += lambda_factor * (reg_component_1 + reg_component_2);

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

for t = 1:m
	% Get t-th training example
	a1_t = X(t, :);
	
	% Compute a2
	z2_t = Theta1 * a1';
	a2_t = sigmoid(z2_t);
	
	% Add bias element to a2_t
	a2_t = [ones(1, columns(a2_t)); a2_t];
	
	% Compute a3
	z3_t = Theta2 * a2_t;
	a3_t = sigmoid(z3_t);
	
	% Get t-th element of y - should be a 10 x 1 vector
	y_t = y(t, :)';
	
	% Compute d3 (for output layer)
	d3 = a3_t - y_t;
	
	% Compute d2 (hidden layer)
	g_z2 = sigmoidGradient(Theta1 * a1');
	g_z2 = [ones(1, columns(g_z2)); g_z2];
	
	d2 = (Theta2' * d3) .* g_z2;
	d2 = d2(2:end); 
	
endfor;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
