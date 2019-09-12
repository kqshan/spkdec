function basis = convert_A_to_spkbasis(self, A)
% Convert the spike basis from the Q1 coordinates back to raw waveforms
%   basis = convert_A_to_spkbasis(self, A)
%
% Returns:
%   basis   [L x C x D] unwhitened spike basis
% Required arguments:
%   A       [L*C x D] whitened spike basis in Q1 coordinates

% A = wh_01 * basis
basis = self.whbasis.wh_01(:,:) \ A;
L = self.L; C = self.C; D = size(basis,2);
basis = reshape(basis, [L C D]);

end
