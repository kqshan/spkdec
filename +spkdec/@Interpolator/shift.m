function y = shift(self, x, r)
% Shift the given vector by the given sub-sample interpolation index
%   y = shift(self, x, r)
%
% Returns:
%   y       [L x M] shifted vectors
% Required arguments:
%   x       [L x M] input vectors
%   r       Sub-sample interpolation index (1..R)
%
% If this object was created using Interpolator.make_interp(),
% then y will correspond to x shifted forward by (r-1)/R sample.
%
% For example, with
%   obj = spkdec.Interpolator.make_interp(4, 'method','linear')
%                x = [0 0 4 4]'
%   obj.shift(x,1) = [0 0 4 4]'
%   obj.shift(x,2) = [0 1 4 4]'
%   obj.shift(x,3) = [0 2 4 4]'
%   obj.shift(x,4) = [0 3 4 4]'

y = self.shifts(:,:,r) * x;

end
