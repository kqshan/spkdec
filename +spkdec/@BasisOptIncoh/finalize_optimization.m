function [basis, spk, resid] = finalize_optimization(self, A, X)
% Finalize the optimization routine and produce the desired outputs
%   [basis, spk, resid] = finalize_optimization(self, A, X)
%
% Returns:
%   basis   [L x C x D] optimized spike basis waveforms
%   spk     Spikes object (where t is the shift in detected spike time)
%   resid   [L+W-1 x C x N] spike residuals (whitened) after optimization
% Required arguments:
%   A       [L*C x D] whitened spike basis in Q1 coordinates
%   X       Struct with fields:
%     X       [D x N] spikes in feature space
%     r       [N x 1] sub-sample shift index (1..R)
%     s       [N x 1] full-sample shift index (1..S)
%
% This also cleans up the following object-level caches:
%   spk_r, Y, A0, lambda, coh_YYt, coh_L

% Call the parent method
args_out = cell(nargout,1);
[args_out{:}] = finalize_optimization@spkdec.BasisOptimizer(self, A, X);
if nargout >= 1, basis = args_out{1}; end
if nargout >= 2, spk   = args_out{2}; end
if nargout >= 3, resid = args_out{3}; end

% Cleanup the additional caches
self.coh_YYt = []; self.coh_L = [];

end
