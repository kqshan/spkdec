function bands = getGramSeq(self, spk_t, spk_r, varargin)
% Return a banded representation of the Gram matrix for a sequence of lags
%   bands = getGramSeq(self, spk_t, spk_r, ...)
%
% Returns:
%   bands     [D*B x D*N] lower diagonals of the Gram matrix
% Required arguments:
%   spk_t     [N x 1] sorted spike times
%   spk_r     [N x 1] sub-sample spike index
% Optional parameters (key/value pairs) [default]:
%   thresh    Threshold to consider a Gram matrix negligible    [ 0 ]
%
% The output format is the same as the BLAS/LAPACK symmetric banded storage
% format (UPLO='L') and is described in more detail in spkdec.Math.pbsolve().
%
% Another way to interpret this is to consider the [D*N x D*N] sparse matrix G:
%   G = spkdec.Math.symband_to_sparse(bands)
% Then if we let ii = (1:D)+D*(i-1) and jj = (1:D)+D*(j-1), we have:
%   G(ii,jj) == self.getGram(spk_t(j)-spk_t(i), spk_r(i), spk_r(j))
%
% The <thresh> param can be used to reduce the resulting bandwidth (B). It is
% used to determine an effective maximum lag L_eff such that
%   For all lag >= L_eff,
%   max(svd(self.getGram(lag))) < thresh * min(svd(self.getGram(0))
% where max(svd(self.getGram(lag))) is computed using self.lagNorms().
% We then assume that self.getGram(lag) == 0 for any lag >= L_eff.
%
% See also: spkdec.Math.pbsolve, spkdec.Math.symband_to_sparse,
% spkdec.Gramians.lagNorms

% Be tolerant of row vectors
spk_t = spk_t(:); spk_r = spk_r(:);

% Dimensions
N = numel(spk_t);
if isscalar(spk_r), spk_r = spk_r * ones(N,1); end
assert(numel(spk_r)==N, self.errid_dim, ...
    'spk_r and spk_t must be the same length');
assert(issorted(spk_t), self.errid_arg, 'spk_t must be sorted');

% Optional parameters
ip = inputParser();
ip.addParameter('thresh', 0, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Determine the maximum gap to consider as overlap
if prm.thresh > 0
    lag_norms = self.lagNorms(); % Matrix norm is same as max(svd(...))
    norm_thresh = prm.thresh * min(svd(self.getGram(0)));
    dt_max = find(lag_norms > norm_thresh, 1, 'last');
    dt_max = dt_max - 1; % Because lag_norms index 1..L correspond to lag 0..L-1
else
    dt_max = self.L-1;
end

% Identify the overlaps
overlaps = self.find_overlaps(spk_t, spk_r, dt_max);

% Get the resulting Gram matrix in a block-banded representation [D x D x B x N]
blkbands = self.get_gram_for_overlaps(overlaps);

% Convert this to the normal banded representation [D*B x D*N]
bands = self.blkband_to_bands(blkbands);

end
