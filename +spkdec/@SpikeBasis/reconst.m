function spikes = reconst(self, spk, varargin)
% Reconstruct the spike waveforms from the given spike features
%   spikes = reconst(self, spk, ...)
%
% Returns:
%   spikes      [L+W-1 x C x N] whitened spike waveforms
% Required arguments:
%   spk         Detected spike times and features (Spikes object)
% Optional parameters (key/value pairs) [default]:
%   subshift    Apply the sub-sample shift indicated by spk.r       [ true ]

% Optional parameters
ip = inputParser();
ip.addParameter('subshift', true, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Check some dimensions
[D,N] = size(spk.X);
[L,K,C] = size(self.basis);
Lw = L + self.W - 1;
assert(D == K*C, self.errid_dim, 'Feature dimensions must match K * C');

% 1. Multiply by the spike basis -----------------------------------------------

% Separate out the features by channel
spk_X = reshape(spk.X, [K C N]);        % [K x C x N]
spk_X = permute(spk_X, [1 3 2]);        % [K x N x C]

% Reconstruct the raw (unwhitened and unshifted) waveforms
spikes = zeros(L,C,N, 'like',spk_X);    % [L x C x N]
for c = 1:C
    spikes(:,c,:) = reshape(self.basis(:,:,c) * spk_X(:,:,c), [L 1 N]);
end

% 2. Apply the whitening and sub-sample shift ----------------------------------

% Apply the sub-sample shift
if prm.subshift
    spikes = self.interp.shiftArr(spikes, spk.r);
end

% Whiten
spikes = reshape(spikes, [L*C, N]);                         % [L*C x N]
spikes = self.whitener.toMat(L, 'flatten',true) * spikes;   % [Lw*C x N]
spikes = reshape(spikes, [Lw, C, N]);                       % [Lw x C x N]

end
