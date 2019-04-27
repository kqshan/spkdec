function A0 = init_spkbasis(self, K)
% Initialize the spike basis from the given data, using a channelwise SVD
%   A0 = init_spkbasis(self, K)
%
% Returns:
%   A0      [L x K x C] whitened spike basis in Q2 coordinates
% Required arguments:
%   K       Number of basis waveforms per channel
%
% Class properties accessed:
%   Y       [L*C x N x S] spike data in the Q1 basis, for shifts -dt:dt

% Dimensions
L = self.L; C = self.C;
[~,N,S] = size(self.Y);
assert(N > 2*K, self.errid_arg, ['The number of spikes must be >> K ' ...
    'for initialization. Get more spikes or specify a basis_prev']);
% We're going to select the shift of Y that corresponds to dt=0
s_idx = (S-1)/2 + 1;
assert(mod(s_idx,1)==0);

% Initialize the spike basis using a channelwise SVD in Q2 space
A0 = zeros(L, K, C);
for c = 1:C
    % Project the data onto the space explainable by this channel
    % Note that map_21(:,:,c) is orthonormal by construction
    Y_Q2_c = self.whbasis.map_21(:,:,c)' * self.Y(:,:,s_idx);  % [L x N]
    % Perform an SVD (but keep only the U part)
    [U,~,~] = svd(Y_Q2_c, 'econ');
    U = gather(double(U));
    A0(:,:,c) = U(:,1:K); % Columns of U will be sorted by singular value
end

end
