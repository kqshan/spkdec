function Ar = get_shifted_basis(self, A, r)
% Compute the shifted version of the given spike basis
%   Ar = get_shifted_basis(self, A, r)
%
% Returns:
%   Ar      [L*C x D] shifted spike basis in Q1 coordinates
% Required arguments:
%   A       Whitened spike basis. Format depends on self.basis_mode:
%             channel-specific - [L x K x C] in Q2 coordinates
%             omni-channel     - [L*C x D] in Q1 coordinates
%   r       Sub-sample shift index (1..R)

% How we do this depends on the basis_mode
switch self.basis_mode
    case 'channel-specific'
        % Apply the shift and convert from Q2 to Q1 coordinates
        % This is slightly complicated by the storage format of A
        [L,K,C] = size(A);
        Ar = zeros(L*C, K, C);
        for c = 1:C
            Ar(:,:,c) = self.whbasis.map_21r(:,:,c,r) * A(:,:,c);
        end
        Ar = reshape(Ar, [L*C, K*C]);
        
    case 'omni-channel'
        % Apply the shift in Q1 coordinates
        Ar = self.whbasis.shift1r(:,:,r) * A;
        
    otherwise
        error(self.errid_arg, 'Unsupported basis_mode "%s"', self.basis_mode);
end

end
