function x = convT(self, y)
% Perform the transpose convolution
%   x = convT(self, y)
%
% Returns:
%   x       [T-V x K x R x C] convolution output
% Required arguments:
%   y       [T x C] data vector

% Check dims
[T,C] = size(y);
assert(C==self.C, self.errid_dim, 'y must be a [T x C] array');
assert(T > self.V, self.errid_dim, 'T must be > self.V (%d)', self.V);

% Perform the convolution
x = self.toConv().convT(y);

% Reshape x as desired
Tx = size(x,1); K = self.K; R = self.R;
x = reshape(x, [Tx, K, R, C]);

end
