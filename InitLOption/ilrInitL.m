function L = ilrInitL(dim)

L = clrInitL(dim); % dim x dim
V = GSorthonormalBase(dim); % dim x (dim - 1)

L = V' * L; % (dim-1) x dim
% add zeros row in the end
L = [L ; zeros(1, dim)];

end
