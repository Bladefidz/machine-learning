function [error_train, error_val] = ...
  randomLearningCurve(X, y, Xval, yval, lambda)
  
  m = size(X,1)     % the number of training examples
  n = size(Xval,1)  % the number of validation examples
  error_train = zeros(m, 1);
  error_val   = zeros(n, 1);
  
  for i = 1:m
    % create two empty vectors for the Jtrain and Jcv values
    Jtrain = zeros(50, 1);
    Jcv = zeros(50, 1);
    for j = 1:50
      % use 'm' to select 'i' random examples from the training set
      it = randi([1, m]);
      % use 'n' to select 'i' random examples from the validation set
      iv = randi([1, n]);
      % compute theta
      theta = trainLinearReg(X(1:it,:), y(1:it), lambda);
      % compute Jtrain and Jcv and save the values
      Jtrain(i) = linearRegCostFunction(X(1:it,:), y(1:it), theta, 0);
      Jvc(i) = linearRegCostFunction(Xval(1:iv,:), yval(1:iv), theta, 0);
    end
    % compute the mean of the Jtrain vector and save it in error_train(i)
    error_train(i) = mean(Jtrain);
    % compute the mean of the Jcv vector and save it in error_val(i)
    error_val(i) = mean(Jcv);
  end
  
end