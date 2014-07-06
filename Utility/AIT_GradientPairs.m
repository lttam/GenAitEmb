function [gradQ, gradb, val] = AIT_GradientPairs(x, idPair, Q, b, flagNORM)

% INPUT
%  x (dim x n) data
%  idPair (2 x nPair) id of pairs
%  Q: (multiplication of Mahalanobis matrix & Aitchison matrix
%  b: (dim x 1) gradient at point b (vector)
% OUTPUT
%  gradQ (dim x dim): gradient of square AG_Mahalanobis distance w.r.t Q
%  gradB (dim x 1): gradient of square AG_Mahalanobis distance w.r.t b
%  val: sum of distance values.

% flagNORM --> normalized or not !!!

[~, nSample] = size(x);
nPair = size(idPair, 2);

xpb = x + repmat(b, 1, nSample);
invxpb = 1 ./ xpb;
logxpb = log(xpb);

logxpb1 = logxpb(:, idPair(1, :));
logxpb2 = logxpb(:, idPair(2, :));

logXBsYB = logxpb1 - logxpb2;
QlogXY = Q*logXBsYB;

invxpb1 = invxpb(:, idPair(1, :));
invxpb2 = invxpb(:, idPair(2, :));

gradXYvector = invxpb1 - invxpb2;

mGradB = 2 * (QlogXY .* gradXYvector);
gradb = sum(mGradB, 2);

% USING C-MEX
gradQ = SOGD(logXBsYB, 1:nPair, 1:nPair);

if (flagNORM == 1)
    gradQ = gradQ ./ nPair;
    gradb = gradb ./ nPair;
end

if nargout > 2
    val = sum(sum(Q .* gradQ));
end

end
