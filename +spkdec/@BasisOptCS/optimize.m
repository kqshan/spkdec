function [basis, spk, resid] = optimize(self, data, varargin)
% Find a channel-specific spike basis that minimizes the reconstruction error
%   [basis, spk, resid] = optimize(self, data, ...)
%
% Returns:
%   basis       [L x C x D] optimized channel-specific spike basis waveforms
%   spk         Spikes object (where t is the shift in detected spike time)
%   resid       [L+W-1 x C x N] spike residuals (whitened) after optimization
% Required arguments:
%   data        [L+W-1 x C x N] detected spike waveforms (whitened)
% Optional parameters (key/value pairs) [default]:
%   lambda      Proximal regularizer weight                 [ 0 ]
%   basis_prev  Previous basis (for proximal regularizer)   [ none ]
%   D           Number of basis waveforms overall         [defer to basis_prev]
%   zero_pad    [pre,post] #samples of zero-padding to add  [ 0,0 ]
%   spk_r       [N x 1] sub-sample shift for each spike     [ auto ]
%
% This finds the basis waveforms and spike features to solve:
%     minimize    ||data - basis*spk.X||^2 + lambda*||basis-basis_prev||^2
%   subject to    basis is channel-specific
%                 basis is channelwise orthonormal
% where the norms are defined in terms of the whitened inner product. "channel-
% specific" means that basis(:,c,d) == 0 unless ceil(d/K)==c, where K = D/C.
%
% If self.dt_search > 0 and/or self.whbasis.R > 1 (unless `spk_r` is given),
% then this process will also search over available full- and/or sub-sample
% shifts in the detected spike times. This reduces the incentive to represent
% such shifts using the spike basis itself.


% --------------------     Problem description     ------------------------

% This reuses the same optimization framework as in the parent class
% BasisOptimizer, but it needs a little bit of tweaking to support the
% "channel-specific and channelwise orthonormal" constraint.

% First, let's review the problem setup in the parent class.
%   data        [(L+W-1)*C x N] observed whitened spike waveforms
%   wh_00       [(L+W-1)*C x L*C] whitening operation (self.whbasis.wh_00)
%   basis       [L*C x D] spike basis waveforms
%   basis_prev  [L*C x D] previous basis for the regularizer
%   X           [D x N] spikes in feature space
% We then transformed the problem using whbasis.Q1, which defines an orthonormal
% basis for the span of the whitener. Using this change of variables:
%   Y = Q1'*data     [L*C x N] observed waveforms in Q1 coordinates
%   A = wh_01*basis  [L*C x D] whitened spike basis in Q1 coordinates
% And since the whitener wh_00 == Q1*wh_01 and Q1'*Q1 == I, the problem becomes:
%     minimize  ||Y - A*X||^2 + lambda*||A - A0||^2
%   subject to  A is orthonormal

% However, we now need to incorporate the "channel-specific and channelwise
% orthonormal" constraint. For this, we turn to whbasis.Q2, which is another L*C
% dimensional basis for the whitener span, except that Q2 is channe-specific and
% channelwise orthonormal. So if we instead let
%   A = wh_02*basis  [L*C x D] whitened spike basis in Q2 coordinates
% and note that whbasis.map_21 defines a map such that wh_01 == map_21*wh_02,
% our problem becomes
%     minimize  ||Y - map_21*A*X||^2 + lambda*||map_21*(A - A0)||^2
%   subject to  A is block diagonal with orthonormal blocks

% We can simplify this one step further if we assume that (A-A0) is block
% diagonal, which is true as long as the given basis_prev is channel-specific.
% Since Q1 is orthonormal and Q2 is channelwise orthonormal, the C [L x L]
% blocks along the diagonal of map_21'*map_21 are all [L x L] identity matrices.
% Then since (A-A0) is block diagonal, the off-diagonal terms of map_21'*map_21
% are irrelevant because they correspond to zeros in (A-A0) and thus
%   ||map_21*(A-A0)|| == ||A-A0||
% which lets us further simplify the second term of our objective.

% Finally, let us note that we can reuse all of the optimization framework from
% the parent class. It's only the individual steps that need to be altered,
% which we will achieve by overloading them.

% One formatting issue to be aware of: since A is block diagonal, we're only
% going to be explicitly storing its C [L x D/C] diagonal blocks, which means
% that instead of A being an [L*C x D] matrix, it will be [L x D/C x C]. This
% will affect all of the operations involving A or A0.

% Call the superclass method
args_out = cell(nargout,1);
[args_out{:}] = optimize@spkdec.BasisOptimizer(self, data, varargin{:});

% Assign output arguments
if nargout >= 1, basis = args_out{1}; end
if nargout >= 2, spk   = args_out{2}; end
if nargout >= 3, resid = args_out{3}; end

end
