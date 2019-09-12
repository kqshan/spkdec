function A = convert_spkbasis_to_A(self, basis)
% Convert the spike basis from raw waveforms into Q1 coordinates
%   A = convert_spkbasis_to_A(self, basis)
%
% Returns:
%   A       [L*C x D] whitened spike basis in Q1 coordinates
% Required arguments:
%   basis   [L x C x D] unwhitened spike basis

% A = wh_01 * basis
[L, C, D] = size(basis);
assert(L==self.L, C==self.C, self.errid_dim, ...
    'basis_prev must be [L x C x D] with L=%d and C=%d',self.L,self.C);
A = self.whbasis.wh_01(:,:) * reshape(basis, [L*C, D]);

end
