function x = pbsolve_mex(A_bands, b)
% Solve a set of linear equations involving a symmetric positive banded matrix
%   x = pbsolve_mex(A_bands, b)
%
% Returns:
%   x           [N x m] solutions to A*x = b
% Required arguments:
%   A_bands     [D x N] lower diagonals of A (assumed to be symmetric)
%   b           [N x m] data vectors
%
% The MEX file achieves a substantial speedup relative to the native MATLAB
% implementation by calling a specific LAPACK routine for banded PSD matrices,
% rather than the 2-step process of converting the bands into the MATLAB sparse
% representation, then performing a linear solve on the sparse matrix.

% This code is only executed if the .mex file does not exist
x = spkdec.Math.pbsolve_m(A_bands, b);

end
