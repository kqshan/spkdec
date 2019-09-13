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
%   f(A) = 1/2*||shift1r*A*X - Y||^2 + lambda/2*||A-A0||^2 + g(A)
% where shift1r implements the sub-sample shift operator. The parent class
% already computes the gradient with respect to the first two terms, so let's
% start with that.
grad = compute_gradient@spkdec.BasisOptimizer(self, A, X);

% For the last term, we have (where cp = coh_penalty):
%   g(A) = cp/2 * sum_t w_t*||A'*shift_t*Y||^2 / sum_t(w_t)
%        = cp/2 * sum_t trace(A'*shift_t*Y*Y'*shift_t'*A) / sum_t(w_t)
%        = cp/2 * trace(A' * sum_t(shift_t*Y*Y'*shift_t')/sum_t(w_t) * A)
% This middle term doesn't depend on A or X. It's computed during initialization
% and stored in the `coh_YYt` object property. Thus,
%        g(A) = cp/2 * trace(A' * coh_YYt * A)
%             = cp/2 * ||coh_L' * A||^2    <-- coh_L*coh_L' == coh_YYt
%   grad_g(A) = cp * coh_YYt * A
grad = grad + self.coh_penalty * self.coh_YYt * A;

end
