function grad = SumGradientPairs(x, idPair, M, b)

% INPUT
%  x (dim x n) data
%  idPair (2 x nPair) id of pairs
%  M: (multiplication of Mahalanobis matrix & Aitchison matrix
%  b: (dim x 1) gradient at point b (vector)
% OUTPUT
%  mGrad (dim x nPair) gradient of square AG_Mahalanobis distance
%

[dim nSample] = size(x);
nPair = size(idPair, 2);

xpb = x + repmat(b, 1, nSample);
invxpb = 1 ./ xpb;
logxpb = log(xpb);

mGrad = zeros(dim, nPair);
for ii = 1:nPair
   % x(:, idPair(1, ii))
   % x(:, idPair(2, ii))
   
   % vector dim x 1 (column vector)
   logXBsYB = logxpb(:, idPair(1, ii)) - logxpb(:, idPair(2, ii));
   
   % (each row matrix M) x logXBsYB
   % repmat(row_vector, dim x 1)
   % sum over row --> sum(-, 2)
   % --> column vector (RESULT)
   weightVector = 2 * sum((M .* repmat(logXBsYB', dim, 1)), 2);
   
   % column vector
   gradXYvector = invxpb(:, idPair(1, ii)) - invxpb(:, idPair(2, ii));
   
   % result (multiplication element-wise between 2 vector weightVector &
   % gradXYvector
   mGrad(:, ii) = weightVector .* gradXYvector;

end

grad = sum(mGrad, 2);
grad = grad/nPair;

end


