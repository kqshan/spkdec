function y = conv(self, x)
% Perform the forward convolution
%   y = conv(self, x)
%
% Returns:
%   y       [T+V x C] convolution output
% Required arguments:
%   x       [T x D x R] spike features

% Check dimensions
[T, D, R] = size(x);
assert(D==self.D && R==self.R, self.errid_dim, 'x must be a [T x D x R] array');
K = self.K; C = self.C;

% Reshape x to be compatible with our convolver object
x = reshape(x, [T K C R]);      % [T x K x C x R]
x = permute(x, [1 2 4 3]);      % [T x K x R x C]
x = reshape(x, [T, K*R*C]);     % [T x K*R*C]

% Perform the convolution
y = self.toConvCS().conv(x);

end
