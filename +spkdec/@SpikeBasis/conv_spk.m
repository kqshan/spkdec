function y = conv_spk(self, spk, T)
% Perform the forward convolution with a spkdec.Spikes object
%   y = conv_spk(self, spk, T)
%
% Returns:
%   y       [T+V x C] convolution output
% Required arguments:
%   spk     Sparse representation of the spike features (Spikes object)
%   T       Overall feature length (#samples)
%
% A note on spike times:
% To keep things simple, this uses a purely causal convolution. This means that
% the spike times (spk.t) correspond to the start of the whitened waveform,
% rather than what you might consider the spike center.

% Convert the spikes object to a [T x D x R] array
x = spk.toFull(T, self.R);

% Perform the convolution
y = self.conv(x);

end
