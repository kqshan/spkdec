function X = blkband_to_bands(G_blkband)
% Convert a matrix from block-banded to standard banded format
%   G_bands = blkband_to_bands(G_blkband)
%
% Returns:
%   G_bands     [D*B x D*N] lower diagonals of a symmetric matrix (same as the
%               BLAS/LAPACK symmetric banded format, UPLO='L')
% Required arguments:
%   G_blkband   [D x D x B x N] array that can be interpreted as the [D x N]
%               lower diagonals of a symmetric matrix, except that each element
%               is a [D x D] block. The blocks along the diagonal (:,:,1,:) must
%               be symmetric.

% Dimensions
[D, D_, B, N] = size(G_blkband);        % [D x D x B x N]
assert(D==D_);

% Transpose this into a [D*B x D x N] matrix
X = permute(G_blkband, [1 3 2 4]);      % [D x B x D x N]
X = reshape(X, [D*B, D, N]);            % [D*B x D x N]

% So we start with something that looks like
%   A11  A12  C11  C12    <--- assume that A21==A12 and C21==C12
%   A21  A22  C21  C22
%   B11  B12  D11  D12
%   B21  B22  D21  D22
% and we want to turn this into
%   A11  A22  C11  C22
%   A21  B12  C21  D12
%   B11  B22  D11  D22
%   B21   0   D21   0

% The first column in each set is fine as-is
for d = 2:D
    % Each subsequent column needs to be shifted up
    X(1:end+1-d, d, :) = X(d:end, d, :);
    % And then filled in with zeros
    X(end+1-d+1:end, d, :) = 0;
end

% Reshape to the desired output
X = reshape(X, [D*B, D*N]);

end
