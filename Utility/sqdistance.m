function D = sqdistance(A, B, M, P)
% Compute square Euclidean or Mahalanobis distances between all pair of vectors.
%   A: d x n1 data matrix
%   B: d x n2 data matrix
%   M: d x d  Mahalanobis matrix
%   D: n1 x n2 pairwise square distance matrix
% Written by Michael Chen (sth4nth@gmail.com).
%
% Modified by Tam Le (lttamvn@gmail.com)for cholinc
% P = cholinc(sparse(M), 'inf'); or ichol
% Modified chol by sqrtm for semidefinite !!!
%
if nargin == 1
    A = bsxfun(@minus,A,mean(A,2));
    S = full(dot(A,A,1));
    D = bsxfun(@plus,S,S')-full(2*(A'*A));
elseif nargin == 2
    assert(size(A,1)==size(B,1));
    D = bsxfun(@plus,full(dot(B,B,1)),full(dot(A,A,1))')-full(2*(A'*B));
elseif nargin == 3
    assert(size(A,1)==size(B,1));
%     if (min(eig(M)) > 0)
%         R = chol(M);
%     else
%         R = sqrtm(M);
%     end
    
    % ORIGINAL
    R = chol(M);
    
    % MODIFIED
%     R = sqrtm(M);
    
    RA = R*A;
    RB = R*B;
    D = bsxfun(@plus,full(dot(RB,RB,1)),full(dot(RA,RA,1))')-full(2*(RA'*RB));
elseif nargin == 4
    assert(size(A,1)==size(B,1));
    R = P;
    RA = R*A;
    RB = R*B;
    D = bsxfun(@plus,full(dot(RB,RB,1)),full(dot(RA,RA,1))')-full(2*(RA'*RB));
end
