function spikes_raw = unwhiten(self, spikes_wh)
% Find the raw waveforms that best approximate the given whitened spikes
%   spikes_raw = unwhiten(self, spikes_wh)
%
% Returns:
%   spikes_raw    [L x C x N] unwhitened spikes
% Required arguments:
%   spikes_wh     [L+W-1 x C x N] spikes in whitened space
%
% This finds the "unwhitened" spikes that most closely approximate the given
% whitened spikes. This is different from simply truncating the spikes to a
% window of length L.
% See also: spkdec.WhitenerBasis.unwhiten

spikes_raw = self.whbasis.unwhiten(spikes_wh);

end
