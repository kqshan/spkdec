function spmat = symband_to_sparse(lband)
% Convert a symmetric matrix from lower banded storage to a MATLAB sparse matrix
%   spmat = symband_to_sparse(lband)
%
% Returns:
%   spmat   [N x N] sparse symmetric matrix
% Required arguments:
%   lband   [D x N] lower diagonals of a matrix (same as BLAS/LAPACK symmetric
%           banded 'SB' storage format with UPLO='L')
%
% A few notes:
% * lband must be double-precision. MATLAB does not support single-precision
%   sparse matrices.
% * If lband is complex, note that spmat will be symmetric, not Hermitian.
% * gpuArrays are supported.

[D,N] = size(lband);
assert(D <= N);

% It may seem like we want spdiags(), but that ultimately just calls sparse(),
% and it's rather inefficient in how it does so (specifically, it uses a layout
% that requires sparse() to re-sort the indices into column-major order). We can
% speed things up quite a bit by ensuring that sparse() does not require any
% data rearrangement, i.e. by ensuring that the indices are in column-major
% order, and that there are no extra zeros in the data provided.

% It may seem like spdiags is what we want, but it's rather inefficient in that
% it ultimately calls sparse() but does so in a way that requires re-sorting the
% indices. There's also quite a lot of values in lband that are zero.

%     Full matrix  |      lband     |        B       |    i    |    j
%   11  21         |                |  0  21  32  43 | 0 1 2 3 | 1 2 3 4
%   21  22  32     | 11  22  33  44 | 11  22  33  44 | 1 2 3 4 | 1 2 3 4
%       32  33  43 | 21  32  43   0 | 21  32  43   0 | 2 3 4 5 | 1 2 3 4
%           43  44 |
B = zeros(2*D-1, N, 'like',lband);
B(D:end,:) = lband;
for d = 1:D-1
    B(D-d,d+1:end) = B(D+d,1:end-d);
end
% Only call sparse() with the nonzero values
mask = (B ~= 0);
B = B(mask);
[ii,jj] = find(mask);
ii = ii + jj - D;
% Call sparse()
spmat = sparse(ii, jj, B, N, N);

% % MATLAB spdiags() wants the diagonals along the columns
% L = lband.';                 % diagonals [0, ..., -(D-1)]
% % And we'll need to fill in the upper triangle
% U = zeros(N, D-1, 'like',L); % diagonals [1, ..., D-1]
% for d = 1:D-1
%     U(d+1:N, d) = L(1:N-d, d+1); % Add a conj() here if you want Hermitian
% end
% % Call spdiags
% d = [0:-1:-(D-1), 1:D-1]';
% spmat = spdiags([L,U], d, N, N);

end
