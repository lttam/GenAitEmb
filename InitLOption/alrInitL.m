function L = alrInitL(dim)

L = eye(dim);
L(:, end) = -ones(dim, 1);
L(end, end) = 0;

end
