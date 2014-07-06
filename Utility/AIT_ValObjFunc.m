function val = AIT_ValObjFunc(x, idPair,idTube, Q, b, L, theta, lambdaB)

% PAIR !!!
[~, nSample] = size(x);
xpb = x + repmat(b, 1, nSample);
logxpb = log(xpb);

dd = sqdistance(logxpb, logxpb, Q, L);

val1 = diag(dd(idPair(1, :), idPair(2, :)));
val1 = sum(val1);

% TRIPLET !!!
SSdd = diag(dd(idTube(1, :), idTube(2, :)));
DDdd = diag(dd(idTube(1, :), idTube(3, :)));

valHL = 1 + SSdd - DDdd;
idHL = valHL > 0;
val2 = sum(valHL(idHL));

val3 = b'*b;

val = val1 + theta*val2 + lambdaB*val3;

end
