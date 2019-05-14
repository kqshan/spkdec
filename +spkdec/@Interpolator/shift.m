function y = shift(self, x, r, trans)
% Shift the given vector by the given sub-sample interpolation index
%   y = shift(self, x, r, [trans])
%
% Returns:
%   y       [L x N] shifted vectors
% Required arguments:
%   x       [L x N] input vectors
%   r       sub-sample shift index (1..R)
% Optional arguments [default]:
%   trans   Use the transpose shift operation instead       [ false ]
%
% If this object was created using Interpolator.make_interp(),
% then y will correspond to x shifted backward by (r-1)/R sample.
%
% For example, with
%   obj = spkdec.Interpolator.make_interp(4, 4, 'method','linear')
%                x = [0 0 4 4]'
%   obj.shift(x,1) = [0 0 4 4]'
%   obj.shift(x,2) = [0 0 3 4]'
%   obj.shift(x,3) = [0 0 2 4]'
%   obj.shift(x,4) = [0 0 1 4]'
%
% If trans==true, then this uses self.shifts(:,:,r)' instead, which
% (generally) corresponds to a forward shift instead.

if nargin < 4, trans = false; end

shift_mat = self.shifts(:,:,r);
if trans, shift_mat = shift_mat'; end

y = shift_mat * x;

end
