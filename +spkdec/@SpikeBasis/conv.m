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

% Reshape x to be compatible with our convolver object
x = reshape(x, [T, D*R]);

% Perform the convolution
y = self.toConv().conv(x);

end
