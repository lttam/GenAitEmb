function L = pcaInitL3(X, tt)

% similar to version 2 (only sampling !!!)

% [evects,evals] = pca(X)
%
% finds principal components of 
%
% input: 
%  X  dxn matrix (each column is a dx1 input vector)
%  tt: percentage 
%
% copyright by Kilian Q. Weinberger, 2006
% modified by Tam Le @ Kyoto

NMAX = 1000;

[d,N]  = size(X);
if (N >= NMAX)
    NN = randperm(N);
    X = X(:, NN(1:NMAX));
    N = NMAX;
end

maxK = round(d*tt);

X=bsxfun(@minus,X,mean(X,2));
cc = cov(X',1); % compute covariance 
[cvv,cdd] = eig(cc); % compute eignvectors
[~,ii] = sort(diag(-cdd)); % sort according to eigenvalues
L = cvv(:,ii(1:maxK)); % pick maxK leading eigenvectors
L = L';

