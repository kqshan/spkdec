function y = interp(self, x)
% Apply the interpolation to the given vector
%   y = interp(self, x)
%
% Returns:
%   y       [R*L x M] interpolated vectors
% Required arguments:
%   x       [L x M] input vectors
%
% This is equivalent to:
%   for r = 1:R
%       y(R-(r-1):R:end, :) = self.shift(x, r)
%
% If this object was created using Interpolator.make_interp(),
% then y(R:R:end,:) == x

% Dimensions
R = self.R;
[L, M] = size(x);
assert(L==self.L, self.errid_dim, 'x must be [L x M] with L=%d',self.L);

% Perform the shifts
y = zeros(R*L, M, 'like',x);
for r = 1:R
    y(R-(r-1):R:end, :) = self.shift(x, r);
end

end
