function A0 = init_spkbasis(self, D)
% Initialize the spike basis from the given data, using a channelwise SVD
%   A0 = init_spkbasis(self, D)
%
% Returns:
%   A0      [L x D/C x C] whitened channel-specific spike basis in Q2 coordinates
% Required arguments:
%   D       Number of basis waveforms overall
%
% Class properties accessed:
%   Y       [L*C x N x S] spike data in the Q1 basis, for shifts -dt:dt

% Dimensions
L = self.L; C = self.C;
[~,N,S] = size(self.Y);
assert(mod(D,C)==0, self.errid_arg, ...
    'For this channel-specific basis optimizer, D must be divisible by C');
min_N = D/C;
assert(N > 2*min_N, self.errid_arg, ['The number of spikes must be >> D ' ...
    'for initialization. Get more spikes or specify a basis_prev']);

% We're going to select the shift of Y that corresponds to dt=0
s_idx = (S-1)/2 + 1;
assert(mod(s_idx,1)==0);

% Initialize the spike basis using SVD on each channel in Q2 coordinates
K = D/C;
A0 = zeros(L, K, C);
for c = 1:C
    % Project the data onto the space explainable by this channel
    % Note that map_21(:,:,c) is orthonormal by construction
    Y_Q2_c = self.whbasis.map_21(:,:,c)' * self.Y(:,:,s_idx);  % [L x N]
    % Perform an SVD (but keep only the U part)
    [U,~,~] = svd(Y_Q2_c, 'econ');
    U = gather(double(U));
    A0(:,:,c) = U(:,1:K); % Columns of U are sorted by singular value
end
    
end
