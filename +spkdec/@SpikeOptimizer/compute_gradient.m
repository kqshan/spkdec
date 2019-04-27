function grad = compute_gradient(self, A, X)
% Compute the gradient of the objective function with respect to A
%   grad = compute_gradient(self, A, X)
%
% Returns:
%   grad    [L x K x C] gradient with respect to A (in block diagonal format)
% Required arguments:
%   A       [L x K x C] whitened spike basis waveforms in Q2 coordinates
%   X       Output struct from optimize_spk()

% Dimensions and local variables
C = self.C;
R = self.R;
whbasis = self.whbasis;

% Our objective function is
%   f(A) = 1/2*||map_21*A*X - Y||^2 + lambda/2*||map_21*(A-A0)||^2
% and so the gradient is
%   grad = map_21'*(map_21*A*X-Y)*X' + lambda*map_21'*map_21*(A-A0)
%
% A couple simplifications that we can apply:
% * We've already computed ErrXt = (map_21*A*X - Y) * X' in optimize_spk()
% * The gradient may be [L*C x K*C], but we only care about the C [L x K] blocks
%   along its diagonal. Within these blocks, map_21'*map_21 == I (due to how the
%   Q1 and Q2 bases were constructed).
%
% So that brings us to:
%   grad(:,:,c) = map_21(:,:,c)'*ErrXt(:,:,c) + lambda*(A(:,:,c) - A0(:,:,c)

% One complication is that we actually have a different map_21r for each
% sub-sample shift index. We're actually not going to use these shift operators
% in the regularizer term, though, so that part stays as just lambda*(A-A0).

% Start with just the regularizer term
grad = self.lambda * (A - self.A0);

% Add in the gradient of the error term
for c = 1:C
    grad_c = grad(:,:,c);
    for r = 1:R
        grad_c = grad_c + whbasis.map_21r(:,:,c,r)' * X.ErrXt(:,:,c,r);
    end
    grad(:,:,c) = grad_c;
end

end
