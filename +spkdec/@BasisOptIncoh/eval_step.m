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

% To summarize the superclass method, we are trying to determine whether
%   f(A) <= f(Ap) + <grad_f(Ap),A-Ap> + L/2*||A-Ap||^2
% where Ap is `prev_A` and f(A) is our objective function:
%   f(A) = 1/2*||map_21*A*X - Y||^2 + lambda/2*||A-A0||^2
% Due to the quadratic form of f(A), it turns out that
%   f(A) = f(Ap) + <grad_f(Ap),A-Ap> + 1/2*||map_21*(A-Ap)*X||^2 ...
%          + lambda/2*||A-Ap||^2
% and substituting this into our original inequality yields
%   ||map_21*(A-Ap)*X||^2 <= (L-lambda)*||A-Ap||^2.
% These are the `lhs` and `rhs` outputs of the superclass method.
[~, lhs, rhs] = eval_step@spkdec.BasisOptimizer(self, A, prev_A, X);

% We now wish to add an additional term to f(A):
%        g(A) = coh_penalty/2 * ||coh_L'*A||^2
%   grad_g(A) = coh_penalty * coh_L*coh_L' * A
% Similar to f(A) above, we can note that
%   g(A) = g(Ap) + <grad_g(Ap),A-Ap> + coh_penalty/2*||coh_L'*(A-Ap)||^2
% and so we can incorporate this into our overall inequality by simply adding
%   lhs += coh_penalty*||coh_L'*(A-Ap)||^2
delta_A = A - prev_A;
lhs = lhs + self.coh_penalty * sum((self.coh_L'*delta_A).^2, 'all');

% Evaluate the backtracking termination criterion
step_ok = (lhs <= rhs);

end
