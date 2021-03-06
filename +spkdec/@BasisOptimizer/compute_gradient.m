function grad = compute_gradient(self, A, X)
% Compute the gradient of the objective function with respect to A
%   grad = compute_gradient(self, A, X)
%
% Returns:
%   grad    [L*C x D] gradient with respect to A
% Required arguments:
%   A       [L*C x D] whitened spike basis in Q1 coordinates
%   X       Output struct from optimize_spk()

% Our objective function is
%   f(A) = 1/2*||map_21*A*X - Y||^2 + lambda/2*||A - A0||^2
% and so the gradient is
%   grad = map_21'*(map_21*A*X-Y)*X' + lambda*(A-A0)
% We've already computed ErrXt = (map_21*A*X - Y)*X', so this comes out to:
%   grad = map_21'*ErrXt + lambda*(A-A0)
%
% This is slightly complicated by the fact that we have a different map_21r (and
% a different ErrXt) for each sub-sample shift index, which we will sum over.

% Start with just the regularizer term
grad = self.lambda * (A - self.A0);

% Add in the gradient of the error term. We have shift1r instead of map_21r.
for r = 1:self.R
    grad = grad + self.whbasis.shift1r(:,:,r)'*X.ErrXt(:,:,r);
end

end
