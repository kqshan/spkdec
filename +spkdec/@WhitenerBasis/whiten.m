function Y_wh = whiten(self, Y_raw)
% Whiten a waveform of length L
%   Y_wh = whiten(self, Y_raw)
%
% Returns:
%   Y_wh        [L+W-1 x C x N] whitened waveform data
% Required arguments:
%   Y_raw       [L x C x N] waveform data (e.g. spike residuals)

% Dimensions
[L,C,N] = size(Y_raw);
assert(L==self.L && C==self.C, self.errid_dim, 'Y_raw must be [L x C x N]');
Lw = L + self.W - 1;

% Reshape and multiply
Y_wh = reshape(self.wh_00,[Lw*C, L*C]) * reshape(Y_raw,[L*C,N]); % [Lw*C x N]
Y_wh = reshape(Y_wh, [Lw C N]);

end
