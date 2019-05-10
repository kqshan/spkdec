function resid = compute_residual(self, spk)
% Compute the residual with the given spikes
%   resid = compute_residual(self, spk)
%
% Returns:
%   resid   [T+V x C] residual (b - A.conv(spk))
% Required arguments:
%   spk     Detected spikes with features (Spikes object)

T = size(self.At_b,1);
resid = self.b - self.A.conv_spk(spk, T);

end
