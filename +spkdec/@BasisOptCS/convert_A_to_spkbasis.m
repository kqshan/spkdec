function basis = convert_A_to_spkbasis(self, A)
% Convert the spike basis from the Q2 coordinates back to raw waveforms
%   basis = convert_A_to_spkbasis(self, A)
%
% Returns:
%   basis   [L x C x D] unwhitened spike basis
% Required arguments:
%   A       [L x K x C] whitened channel-specific spike basis in Q2 coordinates

% A = wh_02 * basis
[L,K,C] = size(A);
basis = zeros(L, C, K, C);
for c = 1:C
    basis_c = self.whbasis.wh_02(:,:,c) \ A(:,:,c);
    basis(:,c,:,c) = reshape(basis_c, [L 1 K 1]);
end
basis = reshape(basis, [L, C, K*C]);

end
