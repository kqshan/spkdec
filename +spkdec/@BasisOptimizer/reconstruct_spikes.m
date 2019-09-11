function spikes = reconstruct_spikes(self, basis, spk, resid)
% Reconstruct the spike waveforms given the detected spikes and residual
%   spikes = reconstruct_spikes(self, basis, spk, resid)
%
% Returns:
%   spikes      [L+W-1 x C x N] spike waveforms (whitened)
% Required arguments:
%   basis       SpikeBasis object
%   spk         Detected spikes (Spikes object)
%   resid       [L+W-1 x C x N] spike residuals (whitened)

% Check that the dimensions match
[Lw, C, N] = size(resid);
assert(Lw == self.L + self.W-1, self.errid_dim, ...
    'Given spike residual waveforms must have length L+W-1');
assert(C == self.C, self.errid_dim, 'Given residuals must have C channels');
assert(N==spk.N, self.errid_dim, 'Number of spike residuals must match spk.N');

% Check that this SpikeBasis has the same whitener and interpolator as us
obj_equal = @(a,b) (a==b) || isequal(a.saveobj(), b.saveobj());
assert(obj_equal(basis.whitener, self.whbasis.whitener), self.errid_arg, ...
    'Given SpikeBasis whitener must match this BasisOptimizer object');
assert(obj_equal(basis.interp, self.whbasis.interp), self.errid_arg, ...
    'Given SpikeBasis interpolator must match this BasisOptimizer object');

% Reconstruct the spike waveforms (detected spike + residual)
spikes = basis.reconst(spk) + resid;

end
