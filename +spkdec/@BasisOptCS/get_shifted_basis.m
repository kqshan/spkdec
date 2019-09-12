function Ar = get_shifted_basis(self, A, r)
% Compute the shifted version of the given spike basis
%   Ar = get_shifted_basis(self, A, r)
%
% Returns:
%   Ar      [L*C x D] shifted spike basis in Q1 coordinates
% Required arguments:
%   A       [L x K x C] whitened channel-specific spike basis in Q2 coordinates
%   r       Sub-sample shift index (1..R)

% Apply the shift and convert from Q2 to Q1 coordinates
% This is slightly complicated by the storage format of A
[L,K,C] = size(A);
Ar = zeros(L*C, K, C);
for c = 1:C
    Ar(:,:,c) = self.whbasis.map_21r(:,:,c,r) * A(:,:,c);
end
Ar = reshape(Ar, [L*C, K*C]);
        
end
