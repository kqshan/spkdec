function [step_ok, lhs, rhs] = eval_step(self, A, prev_A, X)
% Evaluate a proximal gradient descent step for backtracking
%   [step_ok, lhs, rhs] = eval_step(self, A, prev_A, X)
%
% Returns:
%   step_ok   Whether this satisfies the backtracking termination criterion
%   lhs       Left hand side of the criterion (see code for more specifics)
%   rhs       Right hand side of the criterion
% Required arguments:
%   A         New value of the whitened spike basis waveforms
%   prev_A    Previous value of the whitened spike basis waveforms
%   X         Output struct from optimize_spk()
%
% This uses self.lip (local Lipschitz estimate) and expects it to be the same
% value as when we called A = prox_grad_step(prev_A, grad)
%
% This also uses self.grad_g and expects it to contain the gradient of the
% coherence penalty g(A) evaluated at prev_A.

% To summarize the superclass method, we are trying to determine whether
%   f(A) <= f(Ap) + <grad_f(Ap),A-Ap> + L/2*||A-Ap||^2
% where Ap is `prev_A` and f(A) is our objective function:
%   f(A) = 1/2*||map_21*A*X - Y||^2 + lambda/2*||A-A0||^2
% Due to the quadratic form of f(A), it turns out that
%   f(A) = f(Ap) + <grad_f(Ap),A-Ap> + 1/2*||map_21*(A-Ap)*X||^2 ...
%          + lambda/2*||A-Ap||^2
% and substituting this into our original inequality (and multiplying both sides
% by 2) yields
%   ||map_21*(A-Ap)*X||^2 <= (L-lambda)*||A-Ap||^2.
% These are the `lhs` and `rhs` outputs of the superclass method.
[~, lhs, rhs] = eval_step@spkdec.BasisOptimizer(self, A, prev_A, X);

% We now wish to add an additional term to f(A):
%   f(A) += coh/2 * g(A)
% However, this g(A) isn't just a quadratic form, so we will have to update the
% inequality with (remember that both sides have been multiplied by 2)
%   lhs += coh * g(A)
%   rhs += coh * g(Ap) + 2*coh*<grad_g(Ap),A-Ap>.
% Or equivalently (and potentially better-behaved numerically):
%   lhs += coh * (g(A) - g(Ap))
%   rhs += 2 * coh * <grad_g(Ap), A-Ap>

% Update the left hand side
delta_g = self.compute_coherence_penalty(A, prev_A);
lhs = lhs + self.coh_penalty * delta_g;

% Update the right hand side, noting that self.grad_g = grad_g(Ap)
dotprod = sum(self.grad_g .* (A-prev_A), 'all');
rhs = rhs + 2 * self.coh_penalty * dotprod;

% Evaluate the backtracking termination criterion
step_ok = (lhs <= rhs);

end
