function x_hat = convT_hat(self, y_hat)
% Perform the transpose convolution (and sum over channels) in frequency domain
%   x_hat = convT_hat(self, y_hat)
%
% Returns:
%   x_hat       [N x B x D] convolution output (in frequency domain) (D = K*C)
% Required arguments:
%   y_hat       [N x B x C] input data in frequency domain

% Get the [N x C x D] convolution kernels in frequency domain
[N, B, C] = size(y_hat);
kern_hat = self.get_kernels_hat(N);
D = size(kern_hat,3);

% Multiply by the conjugate
x_hat = y_hat .* reshape(conj(kern_hat), [N 1 C D]);
% Sum over channels
x_hat = sum(x_hat, 3);
x_hat = reshape(x_hat, [N B D]);

end
