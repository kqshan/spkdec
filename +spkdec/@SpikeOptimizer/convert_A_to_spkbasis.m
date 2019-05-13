function basis = convert_A_to_spkbasis(self, A)
% Convert the spike basis from the Q2 coordinates back to raw waveforms
%   basis = convert_A_to_spkbasis(self, A)
%
% Returns:
%   basis   [L x K x C] unwhitened spike basis
% Required arguments:
%   A       [L x K x C] whitened spike basis in Q2 coordinates

[L,K,C] = size(A);
basis = zeros(L, K, C);
for c = 1:C
    basis(:,:,c) = self.whbasis.wh_02(:,:,c) \ A(:,:,c);
end

end
