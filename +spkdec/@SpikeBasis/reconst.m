function spikes = reconst(self, spk, varargin)
% Reconstruct the spike waveforms from the given spike features
%   spikes = reconst(self, spk, ...)
%
% Returns:
%   spikes      [L+W-1 x C x N] whitened (or [L x C x N] non-whitened) waveforms
% Required arguments:
%   spk         Detected spike times and features (Spikes object)
% Optional parameters (key/value pairs) [default]:
%   subshift    Apply the sub-sample shift indicated by spk.r       [ true ]
%   whitened    Apply the whitener specified by this basis          [ true ]

% Optional parameters
ip = inputParser();
ip.addParameter('subshift', true, @isscalar);
ip.addParameter('whitened', true, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Check some dimensions
[D,N] = size(spk.X);
[L,C,D_] = size(self.basis);
assert(D==D_, self.errid_dim, 'Spikes must have D=%d dimensions',D_);

% Multiply by the spike basis
spikes = reshape(self.basis,[L*C, D]) * spk.X;
spikes = reshape(spikes, [L C N]);

% Apply the sub-sample shift
if prm.subshift
    spikes = self.interp.shiftArr(spikes, spk.r);
end

% Whiten
if prm.whitened
    spikes = self.whitener.whiten(spikes, 'bounds','keep');
end

end
