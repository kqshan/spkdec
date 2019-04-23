function x = pbsolve(A_bands, b)
% Solve a set of linear equations involving a symmetric positive banded matrix
%   x = pbsolve(A_bands, b)
%
% Returns:
%   x           [N x m] solutions to A*x = b
% Required arguments:
%   A_bands     [D x N] lower diagonals of A (assumed to be symmetric)
%   b           [N x m] data vectors
%
% A_bands should contain the lower diagonals of A, i.e.
%       | 11  12         |
%   A = | 21  22  23     |   A_bands = | 11  22  33  44 |
%       |     32  33  34 |             | 21  32  43   0 |
%       |         43  44 |
% This is the same as the BLAS symmetric banded (SB) storage format with UPLO=L
%
% Datatypes:
% * Complex inputs are not supported
% * If either A or b is single-precision, then x will be too
% * If either A or b is a gpuArray, then x will be too. The computation will
%   happen on CPU, though (since the MEX version only supports CPU).

% Perform the computation on CPU
x = spkdec.Math.pbsolve_mex(gather(A_bands), gather(b));

% Move it back if either input was a gpuArray
if isa(A_bands,'gpuArray') || isa(b,'gpuArray')
    x = gpuArray(x);
end

end
