function spk_X = solve(self, convT_y, spk_t, spk_r, varargin)
% Solve for the optimal spike features given the spike times
%   spk_X = solve(self, convT_y, spk_t, spk_r, ...)
%
% Returns:
%   spk_X     [K*C x N] optimal spike features
% Required arguments:
%   convT_y   [T x K x R x C] output of self.convT(y)
%   spk_t     [N x 1] spike times (1..T)
%   spk_r     [N x 1] spike sub-sample shift index (1..R)
% Optional parameters (key/value pairs) [default]:
%   thresh    Threshold to consider a Gram matrix negligible    [ 1e-3 ]
%
% The <thresh> param is used to reduce the size of the banded inverse problem
% that we need to solve; see spkdec.Gramians.getGramSeq() for more detail.
%
% See also: spkdec.Gramians.getGramSeq

% Dimensions
[T, K, R, C] = size(convT_y);
N = numel(spk_t);
assert(numel(spk_r)==N, self.errid_dim, ...
    'spk_t and spk_r must be the same length');
% Make them column vectors
spk_t = spk_t(:);
spk_r = spk_r(:);

% Optional params
ip = inputParser();
ip.addParameter('thresh', 1e-3, @isscalar);
ip.parse( varargin{:} );
prm = ip.Resuls;

% We are looking for x that minimizes
%   ||y - A*x||
% which is simply
%   x = (A'*A) \ (A'*y)
% The tricky part is dealing with A, which is a [(T+V)*C x K*C*N] matrix in this
% setup. Fortunately, A has some structure that we can exploit:
% * We can interpret A as selected columns from a [(T+V)*C x T*K*R*C] Toeplitz
%   matrix, which allows us to evaluate A'*y by way of self.convT(y).
% * A is quite sparse and so is (A'*A), which is a banded symmetric positive
%   definite matrix. We can construct a banded representation of this matrix
%   using self.toGram().getGramSeq() and solve it using spkdec.Math.pbsolve()

% A'*y
% This is just an array indexing problem
tr = spk_t + T*K*(spk_r-1);         % [N x 1] index
kc = T*(0:K-1)' + T*K*R*(0:C-1);    % [K x C] offset
kctr = kc(:) + tr';                 % [K*C x N] index
Aty = convT_y(kctr);

% A'*A
AtA_bands = self.gramians.getGramSeq(spk_t, spk_r, 'thresh',prm.thresh);

% Solve x = (A'*A) \ (A'*y)
spk_X = spkdec.Math.pbsolve(AtA_bands, Aty(:));

end
