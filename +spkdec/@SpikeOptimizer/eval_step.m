function step_ok = eval_step(self, A, prev_A, X)
% Evaluate a proximal gradient descent step for backtracking
%   step_ok = eval_step(self, A, prev_A, X)
%
% Returns:
%   step_ok   Whether this satisfies the backtracking termination criterion
% Required arguments:
%   A         New value of the whitened spike basis waveforms
%   prev_A    Previous value of the whitened spike basis waveforms
%   X         Output struct from optimize_spk()
%
% The format of `A` and `prev_A` depend on self.basis_mode:
%   channel-specific - [L x K x C] in Q2 coordinates
%   omni-channel     - [L*C x D] in Q1 coordinates
%
% This uses self.lip (local Lipschitz estimate) and expects it to be the same
% value as when we called A = prox_grad_step(prev_A, grad)

% In general, if we can write our objective function as
%   phi(x) = f(x) + g(x)
% where f is smooth and g has a convenient proximal operator, then our proximal
% gradient descent step involves defining the quadratic approximation
%   J(x;y) = f(y) + <grad_f(y),x-y> + g(x) + L/2*||x-y||^2
% where L is the local Lipschitz estimate. Our proximal gradient descent step is
% simply minimizing this J(x;y):
%   x := argmin_x J(x;y) = prox_{g/L}(y - 1/L*grad_f(y)
% Convergence requires that J(x;y) be an overapproximation of phi(x), i.e. that
% J(x;y) >= phi(x). Interestingly, this doesn't need to hold for all x, just for
% the specific x that gets selected by the minimization.
% So the backtracking termination criterion is:
%   phi(x) <= J(x;y)
% or equivalently (since g(x) shows up on both sides)
%   f(x) <= f(y) + <grad_f(y),x-y> + L/2*||x-y||^2

% In our case, let Ap = prev_A, and this backtracking termination criterion is:
%   f(A) <= f(Ap) + <grad_f(Ap),A-Ap> + L/2*||A-Ap||^2
% where
%        f(A) = 1/2*||map_21*A*X - Z||^2 + lambda/2*||A-A0||^2
%   grad_f(A) = map_21'*(map_21*A*X - Z)*X' + lambda*(A-A0)
% Making these substitutions, we find that
%        f(A) = f(Ap) + <grad_f(Ap),A-Ap> + 1/2*||map_21*(A-Ap)*X||^2 ...
%               + lambda/2*||A-Ap||^2
% and therefore we can rewrite our backtracking termination criterion as simply:
%   1/2*||map_21*(A-Ap)*X||^2 + lambda/2*||A-Ap||^2 <= L/2*||A-Ap||^2
% or equivalently
%   ||map_21*(A-Ap)*X||^2 <= (L-lambda)*||A-Ap||^2

% Compute the right hand side
delta_A = A - prev_A;
rhs = (self.lip - self.lambda) * sum(delta_A.^2, 'all');

% Compute the left hand side
% * Note that if we can find X_cov such that X_cov*X_cov' == X*X', then
%       ||map_21*(A-Ap)*X|| = ||map_21*(A-Ap)*X_cov||
%   which saves us from having to perform a multiplication of size N.
% * Since we have multiple sub-sample shifts, we need to use a different
%   map_21r for each of these shifts.
lhs = 0;
for r = 1:self.R
    % Compute delta_A2r = map_21r(:,:,r) * delta_A
    delta_A2r = self.get_shifted_basis(delta_A, r);
    % Add this to the rest
    lhs = lhs + sum((delta_A2r*X.X_cov(:,:,r)).^2, 'all');
end

% Evaluate the backtracking termination criterion
step_ok = (lhs <= rhs);

end
