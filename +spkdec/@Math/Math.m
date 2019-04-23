% Static class of common math operations
%
% Math methods (Static):
%   pbsolve         - Positive banded matrix solve (mex available)
%   symband_to_sparse - Convert a symmetric banded matrix to sparse format
%   is_reg_max      - Determine which peaks are a regional maximum
%   typeconv        - Convert a datatype based on some flags
%
% Other methods (Static):
%   buildMex        - Compile the MEX files for these routines
%   testMex         - Run unit tests on the MEX routines

classdef Math

methods (Static)
    % MEX-accelerated methods
    x = pbsolve(A_bands, b);
    
    % Other math methods
    spmat = symband_to_sparse(lbands);
    mask = is_reg_max(t, x, r);
    x = typeconv(x, flags);

    % MEX management
    buildMex(varargin);
    testMex(varargin);
end

methods (Static, Access=protected)
    % pbsolve
    x = pbsolve_m(A_bands, b);
    x = pbsolve_mex(A_bands, b);
    test_pbsolve(); % TODO
end

end
