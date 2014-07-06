function Dist = mydistance(X1, X2, L)

% Input:
% X1: dim x N
% X2: dim x n
% L: linear transformation --> Mahalanobis M = L' * L
% (dim x dim)
% b: pseudo-count vector
% (dim x 1)
% Output:
% Dist: N x n (Aitchison embedding for Mahalanobis distances)

% -- pre-embedding before LMNN 
%X1 = sqrt(X1 + repmat(b, 1, size(X1, 2)));
%X2 = sqrt(X2 + repmat(b, 1, size(X2, 2)));

% in 4 parameters for sqdistance, we can ignore para 3
if(nargin < 3)
    Dist = distance(X1, X2);
else
    Dist = sqdistance(X1, X2, [], L);
end

end