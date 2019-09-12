function A = prox_grad_step(self, A, grad)
% Perform the proximal gradient descent step
%   A = prox_grad_step(self, A, grad)
%
% Returns:
%   A       [L x K x C] whitened channel-specific spike basis in Q2 coordinates
% Required arguments:
%   A       [L x K x C] previous value of A
%   grad    [L x K x C] gradient evaluated at A
%
% This uses self.lip (local Lipschitz estimate) so make sure that is updated
% appropriately before calling this method.

% Our storage format already enforces the channel-specific constraint (since we
% only store the C [L x K] blocks along the diagonal), so we just need to
% enforce orthonormality independently for each channel. Since the gradient
% descent step can also be performed independently for each channel, we can just
% call the superclass method.
for c = 1:self.C
    A(:,:,c) = prox_grad_step@spkdec.BasisOptimizer(self,A(:,:,c),grad(:,:,c));
end

end
