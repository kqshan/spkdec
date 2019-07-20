function y_hat = conv_hat(self, x_hat)
% Perform the convolution (and sum over kernels) in frequency domain
%   y_hat = conv_hat(self, x_hat)
%
% Returns:
%   y_hat       [N x B x C] convolution-and-sum output (in freq. domain)
% Required arguments:
%   x_hat       [N x B x D] input data in frequency domain

% Get the [N x C x D] convolution kernels in frequency domain
[N, B, D] = size(x_hat);
kern_hat = self.get_kernels_hat(N);
C = size(kern_hat,2);

% Multiply
y_hat = reshape(x_hat, [N B 1 D]) .* reshape(kern_hat, [N 1 C D]);
% Sum over kernels
y_hat = sum(y_hat, 4);

end
