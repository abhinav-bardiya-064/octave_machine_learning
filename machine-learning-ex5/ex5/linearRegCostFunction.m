function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values

%debug only

%debug only
m = length(y); % number of training examples
grad = zeros(size(theta));
rest=size(grad,1);
% You need to return the following variables correctly 
J = 0;
%J =1/2 of sum from i=1 to m (hoxi-yi)^2+lambda/2m sum (theta)^2
h0xi=X*theta;
J=(1/(2*m))*sum((h0xi-y).^2)+(lambda/(2*m))*sum(theta(2:rest,:).^2);


%grad = (1/(2*m))*sum(h0xi-y).xj+(lambda/m)thetaj
%                    
grad = (1/m)*(X'*(h0xi-y));
grad(2:rest,1)= grad(2:rest,1) + (lambda/m)*theta(2:rest,1);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
