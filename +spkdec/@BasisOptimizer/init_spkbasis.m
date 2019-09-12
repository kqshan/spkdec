function A0 = init_spkbasis(self, D)
% Initialize the spike basis from the given data, using a channelwise SVD
%   A0 = init_spkbasis(self, D)
%
% Returns:
%   A0      [L*C x D] whitened spike basis in Q1 coordinates
% Required arguments:
%   D       Number of basis waveforms overall
%
% Class properties accessed:
%   Y       [L*C x N x S] spike data in the Q1 basis, for shifts -dt:dt

% Dimensions
[~,N,S] = size(self.Y);
min_N = D;
assert(N > 2*min_N, self.errid_arg, ['The number of spikes must be >> D ' ...
    'for initialization. Get more spikes or specify a basis_prev']);

% We're going to select the shift of Y that corresponds to dt=0
s_idx = (S-1)/2 + 1;
assert(mod(s_idx,1)==0);

% Initialize the spike basis using SVD in Q1 coordinates
[U,~,~] = svd(self.Y(:,:,s_idx), 'econ');
A0 = gather(double(U(:,1:D)));
    
end
