load('ex8data1.mat');
[mu sigma2] = estimateGaussian(X);
pval = multivariateGaussian(Xval, mu, sigma2);
stepsize = (max(pval) - min(pval)) / 1000;
pv = sortrows([pval yval], 1);
eps = [];
pred = [];
for epsilon = min(pval):stepsize:max(pval)
  eps = [eps; epsilon];
  pred = [pred; sum(pval<epsilon)];
end