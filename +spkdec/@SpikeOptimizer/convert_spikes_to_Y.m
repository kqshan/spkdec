function Y = convert_spikes_to_Y(self, spikes, is_wh)
% Convert the given spikes into coordinates in the Q1 basis
%   Y = convert_to_Q1(self, spikes, is_wh)
%
% Returns:
%   Y         [L*C x N x S] spikes (+/- dt_search) in the Q1 basis
% Required arguments:
%   spikes    [L+W-1 x C x N] whitened or [L x C x N] unwhitened spikes
%   is_wh     Whether the given spikes are whitened, or [] to auto-infer
%
% S = 2*self.dt_search + 1 and is the number of full-sample data shifts that we
% are going to be searching over during the spike optimization step. The S
% dimension of Y corresponds to shifts of (-dt_search:dt_search).

% Check the given dimensions
L = self.L; C = self.C; Lw = L+self.W-1;
[L_, C_, N] = size(spikes);
assert(C_==C, self.errid_dim, 'spikes must have C=%d channels',C);

% Infer is_wh
if isempty(is_wh)
    assert(self.W > 1, self.errid_arg, ['Since self.W==1, we cannot ' ...
        'automatically determine whether the given spikes are whitened']);
    if L_==L
        is_wh = false;
    elseif L_==Lw
        is_wh = true;
    else
        error(self.errid_dim, 'size(spikes,1) must either be L or L+W-1');
    end
end

% Start by whitening the spikes
whbasis = self.whbasis;
if ~is_wh
    spikes = reshape(spikes, [L*C, N]);
    spikes = reshape(whbasis.wh_00, [Lw*C, L*C]) * spikes;
    spikes = reshape(spikes, [Lw, C, N]);       % [Lw x C x N]
end

% Shift and convert into the Q1 basis
dt_list = (-self.dt_search:self.dt_search);
S = length(dt_list);
Y = zeros(L*C, N, S, 'like',spikes);
for s = 1:S
    dt = dt_list(s);
    % Shift the spikes and the Q1 basis (dt > 0 means detecting the spike later)
    pos = max(0, dt);
    neg = min(0, dt);
    Q1 = whbasis.Q1(1-neg:end-pos,:,:);
    spk = spikes(1+pos:end+neg,:,:);
    
    % Perform the change of coordinates
    Lw_s = Lw - abs(dt);
    Y(:,:,s) = reshape(Q1, [Lw_s*C, L*C])' * reshape(spk, [Lw_s*C, N]);
end

end
