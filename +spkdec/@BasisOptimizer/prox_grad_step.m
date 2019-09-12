function A = prox_grad_step(self, A, grad)
% Perform the proximal gradient descent step
%   A = prox_grad_step(self, A, grad)
%
% Returns:
%   A       [L*C x D] whitened spike basis waveforms in Q1 coordinates
% Required arguments:
%   A       [L*C x D] previous value of A
%   grad    [L*C x D] gradient evaluated at A
%
% This uses self.lip (local Lipschitz estimate) so make sure that is updated
% appropriately before calling this method.

% Perform the step
step_size = 1 / self.lip;
A = A - step_size * grad;

% Project onto the constraint set

% This is a task of finding the nearest orthonormal matrix to A:
%     minimize    norm(B-A,'fro')
%   subject to    B'*B == I
% And we can do this by performing an SVD and setting all singular valuest to 1.
[U,~,V] = svd(A, 'econ');
A = U * V';

end
