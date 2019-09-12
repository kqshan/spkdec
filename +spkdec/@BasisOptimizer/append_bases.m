function A = append_bases(~, A1, A2)
% Append two sets of spike bases (in Q1 coordinates)
%   A = append_bases(self, A1, A2)
%
% Returns:
%   A       [L*C x D1+D2] appended and orthonormalized bases in Q1 coords
% Required arguments:
%   A1      [L*C x D1] whitened spike basis in Q1 coordinates
%   A2      [L*C x D2] whitened spike basis in Q2 coordinates

% Append
A = [A1, A2];

% Orthonormalize
[U,~,V] = svd(A,'econ');
A = U * V';

end
