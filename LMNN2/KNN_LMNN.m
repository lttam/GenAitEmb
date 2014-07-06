function preds = KNN_LMNN(y, X, M, k, Xt)

X = X';
Xt = Xt';
A = M * M';

add1 = 0;
if (min(y) == 0),
    y = y + 1;
    add1 = 1;
end
[n,m] = size(X);
[nt, m] = size(Xt);

D = sqdistance(X', Xt', A, M);
[V, Inds] = sort(D);

preds = zeros(k, nt);

for (i=1:nt),
    counts = [];
    for (j=1:k),        
        if (y(Inds(j,i)) > length(counts)),
            counts(y(Inds(j,i))) = 1;
        else
            counts(y(Inds(j,i))) = counts(y(Inds(j,i))) + 1;
        end
        
        [v, preds(j, i)] = max(counts);
    end
end
if (add1 == 1),
    preds = preds - 1;
end
