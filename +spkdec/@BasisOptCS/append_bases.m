function A = append_bases(~, A1, A2)
% Append two sets of spike bases (in Q2 coordinates)
%   A = append_bases(self, A1, A2)
%
% Returns:
%   A       [L x K1+K2 x C] appended and channelwise-orthonormalized bases
% Required arguments:
%   A1      [L x K1 x C] whitened channel-specific spike basis in Q2 coords
%   A2      [L x K2 x C] whitened channel-specific spike basis in Q2 coords

% Append
A = cat(2, A1, A2);

% Orthonormalize
for c = 1:size(A,3)
    [U,~,V] = svd(A(:,:,c),'econ');
    A(:,:,c) = U * V';
end

end
