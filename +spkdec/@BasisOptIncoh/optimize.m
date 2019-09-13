function [basis, spk, resid] = optimize(self, data, varargin)
% Find a spike waveform basis that minimizes our regularized objective function
%   [basis, spk, resid] = optimize(self, data, ...)
%
% Returns:
%   basis       [L x C x D] optimized spike basis waveforms
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
%     minimize    f(basis,spk) + lambda*||basis-basis_prev||^2
%   subject to    basis is orthonormal
% where the norms are defined in terms of the whitened inner product. The
% objective f() is defined in the BasisOptIncoh class documentation and is a
% combination of the spike reconstruction error and an coherence-penalizing
% regularizer.
%
% If self.dt_search > 0 and/or self.whbasis.R > 1 (unless `spk_r` is given),
% then this process will also search over available full- and/or sub-sample
% shifts in the detected spike times. This reduces the incentive to represent
% such shifts using the spike basis itself.

% --------------------     Problem description     ------------------------

% We can simply reuse all of the optimization framework from the parent class,
% altering individual steps by overloading them.

% Call the superclass method
args_out = cell(nargout,1);
[args_out{:}] = optimize@spkdec.BasisOptimizer(self, data, varargin{:});

% Assign output arguments
if nargout >= 1, basis = args_out{1}; end
if nargout >= 2, spk   = args_out{2}; end
if nargout >= 3, resid = args_out{3}; end

end
