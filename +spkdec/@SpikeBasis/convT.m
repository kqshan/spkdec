function x = convT(self, y)
% Perform the transpose convolution
%   x = convT(self, y)
%
% Returns:
%   x       [T-V x D x R] convolution output
% Required arguments:
%   y       [T x C] data vector

% Check dims
[T,C] = size(y);
assert(C==self.C, self.errid_dim, 'y must be a [T x C] array');
assert(T > self.V, self.errid_dim, 'T must be > self.V (%d)', self.V);

% Perform the convolution
x = self.toConv().convT(y);         % [T-V x K*R*C]

% Reshape/permute x as desired
Tx = size(x,1); K = self.K; R = self.R; D = K*C;
x = reshape(x, [Tx, K, R, C]);      % [T-V x K x R x C]
x = permute(x, [1 2 4 3]);          % [T-V x K x C x R]
x = reshape(x, [Tx, D, R]);         % [T-V x D x R]

end
