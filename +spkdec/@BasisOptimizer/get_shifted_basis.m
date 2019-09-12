function Ar = get_shifted_basis(self, A, r)
% Compute the shifted version of the given spike basis
%   Ar = get_shifted_basis(self, A, r)
%
% Returns:
%   Ar      [L*C x D] shifted spike basis in Q1 coordinates
% Required arguments:
%   A       [L*C x D] whitened spike basis in Q1 coordinates
%   r       Sub-sample shift index (1..R)

% Apply the shift in Q1 coordinates
Ar = self.whbasis.shift1r(:,:,r) * A;

end
