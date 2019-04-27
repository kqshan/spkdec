function A = prox_grad_step(self, A, grad)
% Perform the proximal gradient descent step
%   A = prox_grad_step(self, A, grad)
%
% Returns:
%   A       [L x K x C] whitened spike basis waveforms (in Q2 coordinates)
% Required arguments:
%   A       [L x K x C] previous value of A
%   grad    [L x K x C] gradient evaluated at A
%
% This uses self.lip (local Lipschitz estimate) so make sure that is updated
% appropriately before calling this method.

% Perform the step
step_size = 1 / self.lip;
A = A - step_size * grad;

% Project onto the constraint set (for the constraint that A is block diagonal
% with orthonormal blocks). We have already enforced the block diagonal part
% through our storage format (namely, that we are only storing the C [L x K]
% blocks along the diagonal) so we just need to enforce orthonormality.
for c = 1:self.C
    % We don't just want to ensure that A is orthonormal, but we want to find
    % the orthonormal A that is closest to the given A. We achieve this by
    % computing the SVD and setting the singular values to 1 (or so I'm told).
    [U,~,V] = svd(A(:,:,c), 'econ');
    A(:,:,c) = U * V';
end

end
