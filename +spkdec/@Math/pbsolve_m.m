function x = pbsolve_m(A_bands, b)
% Solve a set of linear equations involving a symmetric positive banded matrix
%   x = pbsolve_m(A_bands, b)
%
% Returns:
%   x           [N x m] solutions to A*x = b
% Required arguments:
%   A_bands     [D x N] lower diagonals of A (assumed to be symmetric)
%   b           [N x m] data vectors
%
% This is the native MATLAB version, and it constructs the sparse matrix A from
% the given banded storage format. As a consequence, all computation will be
% performed in double-precision.

% Construct the sparse matrix
A = spkdec.Math.symband_to_sparse(double(A_bands));

% Perform the linear inverse
x = A \ double(b);

% Convert the data to match
if isa(A_bands,'single') || isa(b,'single') || ...
        (isa(b,'gpuArray') && strcmp(classUnderlying(b),'single')) || ...
        (isa(A_bands,'gpuArray') && strcmp(classUnderlying(A_bands),'single'))
    x = single(x);
end

end
