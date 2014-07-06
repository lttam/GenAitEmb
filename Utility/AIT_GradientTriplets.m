function [gradQ, gradb, val] = AIT_GradientTriplets(x, idTriplet, Q, b, L, flagNORM)

% INPUT
%  x (dim x n) data
%  idTriplet (3 x nPair) id of Triplet (I,J,K) --> (I,J) in S & (I,K) in D
%  Q: (multiplication of Mahalanobis matrix & Aitchison matrix
%  b: (dim x 1) gradient at point b (vector)
% OUTPUT
%  gradQ (dim x dim): gradient of square AG_Mahalanobis distance w.r.t Q
%  gradB (dim x 1): gradient of square AG_Mahalanobis distance w.r.t b
%  val: sum of distance values.

% flagNORM --> normalized or not !!!

[~, nSample] = size(x);
nTriplet = size(idTriplet, 2);

xpb = x + repmat(b, 1, nSample);
invxpb = 1 ./ xpb;
logxpb = log(xpb);

logxpb1 = logxpb(:, idTriplet(1, :));
logxpb2 = logxpb(:, idTriplet(2, :));
logxpb3 = logxpb(:, idTriplet(3, :));

SSlogXBsYB = logxpb1 - logxpb2;
DDlogXBsYB = logxpb1 - logxpb3;

SSQlogXY = Q*SSlogXBsYB;
DDQlogXY = Q*DDlogXBsYB;

invxpb1 = invxpb(:, idTriplet(1, :));
invxpb2 = invxpb(:, idTriplet(2, :));
invxpb3 = invxpb(:, idTriplet(3, :));

SSgradXYvector = invxpb1 - invxpb2;
DDgradXYvector = invxpb1 - invxpb3;

SSmGradB = 2 * (SSQlogXY .* SSgradXYvector);
DDmGradB = 2 * (DDQlogXY .* DDgradXYvector);

% DISTANCE for ALL
dd = sqdistance(logxpb, logxpb, Q, L);
SSdd = diag(dd(idTriplet(1, :), idTriplet(2, :)));
DDdd = diag(dd(idTriplet(1, :), idTriplet(3, :)));
valHL = 1 + SSdd - DDdd;
idHL = valHL > 0;

disp(['Hinge Loss : ' num2str(sum(idHL)) '/' num2str(nTriplet)]);

% DEPEND on HINGE LOSS !!! (1 + dS - dD)+
gradb = sum(SSmGradB(:, idHL), 2) - sum(DDmGradB(:, idHL), 2);

activeHL = find(valHL > 0);

% USING C-MEX
gradQ1 = SOGD(SSlogXBsYB, activeHL, activeHL);
gradQ2 = SOGD(DDlogXBsYB, activeHL, activeHL);
gradQ = gradQ1 - gradQ2;

if (flagNORM == 1)
    gradQ = gradQ ./ nTriplet;
    gradb = gradb ./ nTriplet;
end

if nargout > 2
    val = sum(sum(Q .* gradQ)) + sum(idHL);
end

end
  
