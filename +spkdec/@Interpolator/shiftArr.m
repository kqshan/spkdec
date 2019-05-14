function X = shiftArr(self, X, r, trans)
% Apply the selected shifts to an [L x C x N] array
%   Y = shiftArr(self, X, r, [trans])
%
% Returns:
%   Y       [L x C x N] shifted data
% Required arguments:
%   X       [L x C x N] input data
%   r       [N x 1] sub-sample shift index (1..R)
% Optional arguments [default]:
%   trans   Use the transpose shift operation instead       [ false ]
%
% See also: spkdec.Interpolator.shift

if nargin < 4, trans = false; end

% Check the dimensions
[L,C,N] = size(X);
assert(N==numel(r), self.errid_dim, 'The length of r must match X');

% Group by shift index
R = self.R;
r_idx = accumarray(r(:), (1:N)', [R 1], @(x) {x});
for r = 1:R
    idx = r_idx{r};
    Nr = length(idx);
    if (Nr==0), continue; end
    % Need to do some reshaping to perform the shift
    y = X(:,:,idx);                 % [L x C x N]
    y = reshape(y, [L, C*Nr]);      % [L x C*N]
    y = self.shift(y, r, trans);    % [L x C*N], shifted
    y = reshape(y, [L C Nr]);       % [L x C x N]
    % We can save the result in the same matrix
    X(:,:,idx) = y;
end

end
