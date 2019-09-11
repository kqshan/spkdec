function A0 = init_spkbasis(self, D)
% Initialize the spike basis from the given data, using a channelwise SVD
%   A0 = init_spkbasis(self, D)
%
% Returns:
%   A0      Whitened spike basis. Format depends on self.basis_mode:
%             channel-specific - [L x D/C x C] in Q2 coordinates
%             omni-channel     - [L*C x D] in Q1 coordinates
% Required arguments:
%   D       Number of basis waveforms overall
%
% Class properties accessed:
%   Y       [L*C x N x S] spike data in the Q1 basis, for shifts -dt:dt

% Dimensions
L = self.L; C = self.C;
[~,N,S] = size(self.Y);
if strcmp(self.basis_mode,'channel-specific')
    assert(mod(D,C)==0, self.errid_arg, ['In the "channel-specific" ' ...
        'basis_mode, D must be divisible by C']);
    min_N = D/C;
else
    min_N = D;
end
assert(N > 2*min_N, self.errid_arg, ['The number of spikes must be >> D ' ...
    'for initialization. Get more spikes or specify a basis_prev']);

% We're going to select the shift of Y that corresponds to dt=0
s_idx = (S-1)/2 + 1;
assert(mod(s_idx,1)==0);

% Initialize the spike basis using SVD
switch (self.basis_mode)
    case 'channel-specific'
        % SVD on each channel in Q2 coordinates
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
        
    case 'omni-channel'
        % SVD in Q1 coordinates
        [U,~,~] = svd(self.Y(:,:,s_idx), 'econ');
        A0 = gather(double(U(:,1:D)));
        
    otherwise
        error(self.errid_arg, 'Unsupported basis_mode "%s"',self.basis_mode);
end
    
end
