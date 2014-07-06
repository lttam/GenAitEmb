function V = GSorthonormalBase(dim)
%
% FIND an orthonormal base for a subspace
% in R^dim s.t norm-L1 = 0
%
% (ILR): isometric logratio
%

A = eye(dim);
A(1, 2:dim-1) = (-1)*ones(1,dim-2);
A(dim, 1) = -1;
A = A(:, 1:dim-1);

V = GramSchmidt(A);

end