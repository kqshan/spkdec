function Y = convert_spikes_to_Y(self, spikes, pad)
% Convert the given spikes into coordinates in the Q1 basis
%   Y = convert_to_Q1(self, spikes, pad)
%
% Returns:
%   Y         [L*C x N x S] spikes (+/- dt_search) in the Q1 basis
% Required arguments:
%   spikes    [L+W-1-sum(pad) x C x N] whitened spike waveforms
%   pad       [pre,post] amount (#samples) to zero-pad the spikes with
%
% S = 2*self.dt_search + 1 and is the number of full-sample data shifts that we
% are going to be searching over during the spike optimization step. The S
% dimension of Y corresponds to shifts of (-dt_search:dt_search).

% Check the given dimensions
L = self.L; C = self.C; Lw = L+self.W-1;
[Lw_, C_, N] = size(spikes);
assert(Lw_==Lw-pad(1)-pad(2), self.errid_dim, ...
    'spikes must have length L+W-1-sum(pad)=%d',Lw-pad(1)-pad(2));
assert(C_==C, self.errid_dim, 'spikes must have C=%d channels',C);
whbasis = self.whbasis;

% Shift and convert into the Q1 basis
dt_list = (-self.dt_search:self.dt_search);
S = length(dt_list);
Y = zeros(L*C, N, S, 'like',spikes);
for s = 1:S
    dt = dt_list(s);
    % Shift the spikes and the Q1 basis (dt > 0 means detecting the spike later)
    Q_offset = max(0, pad(1)-dt);
    Lw_s = min(Lw, Lw-pad(2)-dt) - Q_offset;
    Q1 = whbasis.Q1(Q_offset+(1:Lw_s), :, :);
    spk_offset = Q_offset + dt - pad(1);
    spk = spikes(spk_offset+(1:Lw_s), :, :);
    
    % Perform the change of coordinates
    Y(:,:,s) = reshape(Q1, [Lw_s*C, L*C])' * reshape(spk, [Lw_s*C, N]);
end

end
