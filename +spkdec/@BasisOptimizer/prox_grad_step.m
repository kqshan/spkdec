function A = prox_grad_step(self, A, grad)
% Perform the proximal gradient descent step
%   A = prox_grad_step(self, A, grad)
%
% Returns:
%   A       Whitened spike basis waveforms
% Required arguments:
%   A       Previous value of A
%   grad    Gradient evaluated at A
%
% The format of `A` and `grad` depend on self.basis_mode:
%   channel-specific - [L x K x C] in Q2 coordinates
%   omni-channel     - [L*C x D] in Q1 coordinates
%
% This uses self.lip (local Lipschitz estimate) so make sure that is updated
% appropriately before calling this method.

% Perform the step
step_size = 1 / self.lip;
A = A - step_size * grad;

% Project onto the constraint set
switch (self.basis_mode)
    case 'channel-specific'
        % We have already enforced the block diagonal constraint through our
        % storage format (namely, that we only store the C [L x K] blocks from
        % the diagonal). We just need to enforce orthonormality independently
        % for each channel.
        for c = 1:self.C
            A(:,:,c) = nearest_orthonormal_matrix(A(:,:,c));
        end
        
    case 'omni-channel'
        % Our only constraint is orthonormality of the full [L*C x D] basis
        A = nearest_orthonormal_matrix(A);
        
    otherwise
        error(self.errid_arg, 'Unsupported basis_mode "%s"',prm.basis_mode);
end

end


% ---------------------------     Helper functions     -------------------------

function B = nearest_orthonormal_matrix(A)
% Find the nearest (in terms of Frobenius norm) orthonormal matrix to A
%   B = nearest_orthonormal_matrix(A)
%
% Returns:
%   B       [N x M] matrix with orthonormal columns (A'*A == I)
% Required arguments:
%   A       [N x M] matrix (N >= M)
%
% This minimizes norm(B-A,'fro') among all B such that B'*B == I.
[U,~,V] = svd(A, 'econ');
B = U * V'; % Equivalent to setting all singular values to 1
end
