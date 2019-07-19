function y = conv(self, x)
% Perform the forward convolution
%   y = conv(self, x)
%
% Returns:
%   y       [T+V x C] convolution output
% Required arguments:
%   x       [T x K x R x C] spike features

% Check dimensions
[T, K, R, C] = size(x);
assert(K==self.K && R==self.R && C==self.C, self.errid_dim, ...
    'x must be a [T x K x R x C] array');

% Reshape x to be compatible with our convolver object
x = reshape(x, [T, K*R*C]);

% Perform the convolution
y = self.toConv().conv(x);

end
