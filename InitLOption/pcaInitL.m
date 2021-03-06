function L = pcaInitL(X, pp)
% [evects,evals] = pca(X)
%
% finds principal components of 
%
% input: 
%  X  dxn matrix (each column is a dx1 input vector)
%  pp(propotional/ratio): kept d/pp 
%
% copyright by Kilian Q. Weinberger, 2006
% modified by Tam Le @ Kyoto

[d,N]  = size(X);
maxK = round(d/pp);

X=bsxfun(@minus,X,mean(X,2));
cc = cov(X',1); % compute covariance 
[cvv,cdd] = eig(cc); % compute eignvectors
[~,ii] = sort(diag(-cdd)); % sort according to eigenvalues
L = cvv(:,ii(1:maxK)); % pick maxK leading eigenvectors
L = L';

