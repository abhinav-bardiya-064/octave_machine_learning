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

z=X*theta;
%abhi j=J0=(-1/m)sum from 1 to m (yilog(sigmoid(z))+(1-yi)log(1-sigmoid(z))) + 1/2m sum oj from 1 to n
rest=size(grad,1);
J=(-1/m)*(y'*log(sigmoid(z))+(1-y)'*log(1-sigmoid(z))) + (lambda/(2*m)) * sum(theta([2:rest],1).^2);

%grad1 = (1/m)(sum from 1 to m (sigmoid(z)-yi)xij)

grad=(1/m)*(X'*(sigmoid(z)-y));

%grad2 and above =(1/m)(sum from 1 to m (sigmoid(z)-yi)xij) + lmda/m oj

grad([2:rest],1) = grad ([2:rest],1) + (lambda/m)* theta([2:rest],1);


% =============================================================

end
