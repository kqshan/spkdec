function grad = compute_gradient(self, A, X)
% Compute the gradient of the objective function with respect to A
%   grad = compute_gradient(self, A, X)
%
% Returns:
%   grad    [L*C x D] gradient with respect to A
% Required arguments:
%   A       [L*C x D] whitened spike basis in Q1 coordinates
%   X       Output struct from optimize_spk()
%
% This also updates the local cache self.grad_g

% Our objective function is (where coh = coh_penalty)
%   f(A) = 1/2*||shift1r*A*X - Y||^2 + lambda/2*||A-A0||^2 + coh/2*g(A)
% where shift1r implements the sub-sample shift operator. The parent class
% already computes the gradient with respect to the first two terms, so let's
% start with that.
grad = compute_gradient@spkdec.BasisOptimizer(self, A, X);

% Some dimensions and local variables
R = self.R;
LC = size(A,1);
whbasis = self.whbasis;

% For the last term, we have (see the comments in optimize.m):
%   grad_g(A) = 1/R * sum_r [ 2 * (I - Sr'*Sr*A*inv(Mr)*A') * 
%                             (Sr'*B*Sr*A*inv(Mr)*A' + [...]') * A*inv(Mr) ]
% where
%   Sr = shift matrix (in Q1 coordinates) for a sub-sample shift of r
%   Mr = A'*Sr'*Sr*A
%   B  = 1/T * sum_t Q1'*shift_t*data*data'*shift_t'*Q1 = coh_YYt

% NB: Because of the way that our Lipschitz step sizes is defined, we're
% actually returing 1/2*gradient, so we're dropping the factor of 2 in our
% gradient computation. This was also done in the superclass method.

grad_g = zeros(size(A));
for r = 1:R
    % Evaluate the summand
    Sr = whbasis.shift1r(:,:,r);
    % Start by computing
    %   SBSAMA = Sr'*B*Sr*A*inv(Mr)*A'.
    % We can simplify this by performing the QR decomposition of Sr*A:
    %   Qa * Ra = Sr * A.
    % Since Mr = A'*Sr'*Sr*A = Ra'*Ra, we therefore have
    %   Sr*A*inv(Mr) = Qa * inv(Ra)'
    %   SBSAMA = Sr'*B*Qa*inv(Ra)'*A'
    [Qa,Ra] = qr(Sr*A, 0);
    ARinv = A / Ra;
    SAMA = Qa * ARinv';                 % = Sr*A*inv(Mr)*A'
    SBSAMA = Sr' * self.coh_YYt * SAMA; % = Sr'*B*Sr*A*inv(Mr)*A'
    % We can then break up the summand into
    %   summand = prescale * grad_fixed
    % where
    %   prescale = (I - Sr'*Sr*A*inv(Mr)*A')
    %   grad_fixed = (Sr'*B*Sr*A*inv(Mr)*A' + [...]') * A*inv(Mr)
    % Just FYI, `grad_fixed` is what the gradient of g(A) would be if we treated
    % the Mr as a constant.
    prescale = eye(LC) - Sr'*SAMA;
    grad_fixed = (SBSAMA + SBSAMA') * ARinv / Ra';
    % Sum over r
    grad_g = grad_g + prescale * grad_fixed / R;
end

% Add this last term to the overall gradient
grad = grad + self.coh_penalty * grad_g;

% Also save this in the local cache
self.grad_g = grad_g;

end
