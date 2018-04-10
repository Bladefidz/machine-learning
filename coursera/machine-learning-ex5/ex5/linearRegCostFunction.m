function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
  %regression with multiple variables
  %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
  %   cost of using theta as the parameter for linear regression to fit the 
  %   data points in X and y. Returns the cost in J and the gradient in grad

  % Initialize some useful values
  m = length(y); % number of training examples

  % You need to return the following variables correctly 
  J = 0;
  grad = zeros(size(theta));
  R = theta';
  R(1, 1) = 0;  % Theta0 should not regularized

  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost and gradient of regularized linear 
  %               regression for a particular choice of theta.
  %
  %               You should set J to the cost and grad to the gradient.
  %
  
  h = (X*theta) - y;
  J = ( sum(h.^2)  + lambda*sum(R.^2) ) / (2*m);
  grad = ( sum(h.*X) + lambda.*R ) / m;
  
  % =========================================================================

  grad = grad(:);

end
