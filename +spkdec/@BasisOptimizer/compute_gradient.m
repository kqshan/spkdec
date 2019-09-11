function grad = compute_gradient(self, A, X)
% Compute the gradient of the objective function with respect to A
%   grad = compute_gradient(self, A, X)
%
% Returns:
%   grad    Gradient with respect to A (same format as A)
% Required arguments:
%   A       Whitened spike basis. Format depends on self.basis_mode:
%             channel-specific - [L x K x C] in Q2 coordinates
%             omni-channel     - [L*C x D] in Q1 coordinates
%   X       Output struct from optimize_spk()

% Dimensions and local variables
[LC,D,R] = size(X.ErrXt);
C = self.C;
whbasis = self.whbasis;

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

% Add in the gradient of the error term
switch self.basis_mode
    case 'channel-specific'
        % We can do this independently for each channel since we don't care
        % about the off-diagonal blocks
        K = D/C;
        ErrXt = reshape(X.ErrXt, [LC, K, C, R]);
        for c = 1:C
            for r = 1:R
                grad_cr = whbasis.map_21r(:,:,c,r)' * ErrXt(:,:,c,r);
                grad(:,:,c) = grad(:,:,c) + grad_cr;
            end
        end
        
    case 'omni-channel'
        % In this case, we have shift1r instead of map_21r.
        for r = 1:R
            grad = grad + whbasis.shift1r(:,:,r)'*X.ErrXt(:,:,r);
        end
        
    otherwise
        error(self.errid_arg, 'Unsupported basis_mode "%s"',self.basis_mode);
end

end
