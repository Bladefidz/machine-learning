function [C, sigma] = dataset3Params(X, y, Xval, yval)
  %DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
  %where you select the optimal (C, sigma) learning parameters to use for SVM
  %with RBF kernel
  %   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
  %   sigma. You should complete this function to return the optimal C and 
  %   sigma based on a cross-validation set.
  %

  % You need to return the following variables correctly.
  C = 1;
  sigma = 0.1;
%  model = svmTrain(Xval, yval, C, ...
%        @(x1, x2) gaussianKernel(x1, x2, sigma));
%  predictions = svmPredict(model, Xval);
%  error = mean(double(predictions ~= yval))

  % ====================== YOUR CODE HERE ======================
  % Instructions: Fill in this function to return the optimal C and sigma
  %               learning parameters found using the cross validation set.
  %               You can use svmPredict to predict the labels on the cross
  %               validation set. For example, 
  %                   predictions = svmPredict(model, Xval);
  %               will return the predictions on the cross validation set.
  %
  %  Note: You can compute the prediction error using 
  %        mean(double(predictions ~= yval))
  %
  
%  params = [0.01 0.03 0.1 0.3 1 3 10 30];
%  m = length(params);
%  minError = inf;
%  for i=1:m
%    tmpC = params(i);
%    for j=1:m
%      tmpSigma = params(j);
%      model = svmTrain(X, y, tmpC, ...
%        @(x1, x2) gaussianKernel(x1, x2, tmpSigma));
%      predictions = svmPredict(model, Xval);
%      error = mean(double(predictions ~= yval));
%      fprintf("Loop i=%d j=%d ", i, j);
%      if (error < minError)
%        C = tmpC;
%        sigma = tmpSigma;
%        minError = error;
%        fprintf("Min ");
%      end
%      fprintf("C=%f, sigma=%f, error=%f\n", tmpC, tmpSigma, error);
%    end
%  end

%  hi = length(params);
%  lo = 0;
%  while (lo < hi)
%    mid = floor(lo + (hi - lo) / 2);
%    tmpC = params(mid);
%    model = svmTrain(Xval, yval, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%    predictions = svmPredict(model, Xval);
%    error = mean(double(predictions ~= yval));
%    if (error < minError)
%      minError = error;
%      C = tmpC;
%      hi = mid - 1;
%    elseif (error > minError)
%      lo = mid + 1;
%    end
%  endwhile
%  while (lo < hi)
%    mid = floor(lo + (hi - lo) / 2);
%    tmpSigma = params(mid);
%    model = svmTrain(Xval, yval, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%    predictions = svmPredict(model, Xval);
%    error = mean(double(predictions ~= yval));
%    if (error < minError)
%      minError = error;
%      sigma = tmpSigma;
%      hi = mid - 1;
%    elseif (error > minError)
%      lo = mid + 1;
%    end
%  endwhile

  % =========================================================================
end
