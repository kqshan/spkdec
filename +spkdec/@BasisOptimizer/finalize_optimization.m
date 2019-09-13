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
% This also cleans up the following object-level caches: spk_r, Y, A0, lambda

% Convert the basis back into raw waveforms
basis = self.convert_A_to_spkbasis(A);

% Construct the Spikes object
spk_t = X.s - (self.dt_search+1);
spk = spkdec.Spikes(spk_t, X.r, X.X);

% Compute the residual
if nargout >= 3
    resid = self.eval_error(A, X);  % [L*C x N] in Q1 coordinates
    L = self.L; C = self.C; Lw = L + self.W-1;
    resid = reshape(self.whbasis.Q1,[Lw*C,L*C]) * resid;
    resid = reshape(resid, [Lw, C, spk.N]);
end

% Cleanup
self.spk_r = []; self.Y = []; self.A0 = []; self.lambda = [];

end
