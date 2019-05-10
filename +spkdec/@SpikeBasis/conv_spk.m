function y = conv_spk(self, spk, T)
% Perform the forward convolution with a spkdec.Spikes object
%   y = conv_spk(self, spk, T)
%
% Returns:
%   y       [T+V x C] convolution output
% Required arguments:
%   spk     Sparse representation of the spike features (Spikes object)
%   T       Overall feature length (#samples)

% Convert the spikes object to a [T x K x R x C] array
x = spk.toFull(T, self);

% Perform the convolution
y = self.conv(x);

end
