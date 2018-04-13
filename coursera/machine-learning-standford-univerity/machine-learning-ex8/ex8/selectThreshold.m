function [bestEpsilon bestF1] = selectThreshold(yval, pval)
  %SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
  %outliers
  %   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
  %   threshold to use for selecting outliers based on the results from a
  %   validation set (pval) and the ground truth (yval).
  %

  bestEpsilon = 0;
  bestF1 = 0;
  F1 = 0;
  pv = sortrows([pval yval], 1);
  yval0 = pv(1:end, 2) == 0;
  yval1 = pv(1:end, 2) == 1;
  pv = pv(1:end, 1);
  pless = zeros(length(pv));
  pmore = ones(length(pv));
  i = 0;
  j = 1;
  spl = 0;

  stepsize = (max(pval) - min(pval)) / 1000;  % This means we loop 1001 times
  for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    
%    if i == 0
%      continue;
%    endif
%    if pv(i++) >= epsilon
%      continue;
%    endif
%    
%    pless(j:i) = 1;
%    pmore(j:i) = 0;
%    j = i+1;
    pless = pv < epsilon;
    sp = sum(pless);
    if sp == spl
      continue
    end
    spl = sp;
    pmore = pv >= epsilon;
    
    fp = sum(pless & yval0);
    tp = sum(pless & yval1);
    fn = sum(pmore & yval1);
    prec = tp / (tp + fp);
    rec = tp / (tp + fn);
    F1 = (2*prec*rec) / (prec+rec);
    
    % =============================================================

    if F1 > bestF1
      bestF1 = F1;
      bestEpsilon = epsilon;
    end
  end

end
