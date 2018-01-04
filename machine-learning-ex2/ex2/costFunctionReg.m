function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


theta_without_0 = theta(2:end);

h = sigmoid(theta' * X');
J = (1 / m) * ((log(h) * -y) - (log(1 - h) * (1 - y)));
reg_param = (lambda / (2 * m)) * (theta_without_0' * theta_without_0);
J = J + reg_param;

e = h' - y;

x0 = X(:, 1);
grad(1) = (1 / m) * (e' * x0);

for i = 2:length(grad)
	xi = X(:, i);
	grad(i) = (1 / m) * (e' * xi) + (lambda / m) * theta(i);
endfor

% =============================================================

end
