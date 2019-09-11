function basis = convert_A_to_spkbasis(self, A)
% Convert the spike basis from the Q2 coordinates back to raw waveforms
%   basis = convert_A_to_spkbasis(self, A)
%
% Returns:
%   basis   Unwhitened spike basis. Format depends on self.basis_mode:
%             channel-specific - [L x K x C]
%             omni-channel     - [L x C x D]
% Required arguments:
%   A       Whitened spike basis. Format depends on self.basis_mode:
%             channel-specific - [L x K x C] in Q2 coordinates
%             omni-channel     - [L*C x D] in Q1 coordinates

switch (self.basis_mode)
    case 'channel-specific'
        % A = wh_02 * basis
        [L,K,C] = size(A);
        basis = zeros(L, K, C);
        for c = 1:C
            basis(:,:,c) = self.whbasis.wh_02(:,:,c) \ A(:,:,c);
        end
        
    case 'omni-channel'
        % A = wh_01 * basis
        basis = self.whbasis.wh_01(:,:) \ A;
        L = self.L; C = self.C; D = size(basis,2);
        basis = reshape(basis, [L C D]);
        
    otherwise
        error(self.errid_arg, 'Unsupported basis_mode "%s"',self.basis_mode);
end

end
