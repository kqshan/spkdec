function x_hat = convT_hat(y_hat, kern_hat)
% Perform the transpose convolution in frequency domain
%   x_hat = convT_hat(y_hat, kern_hat)
%
% Returns:
%   x_hat       [N x B x D] convolution output (in frequency domain) (D = K*C)
% Required arguments:
%   y_hat       [N x B x C] input data in frequency domain
%   kern_hat    [N x K x C] convolution kernels in frequency domain

% Dimensions
[N, B, C] = size(y_hat);
K = size(kern_hat,2);

% Multiply by the conjugate
conj_kern_hat = conj(kern_hat);
x_hat = reshape(y_hat, [N B 1 C]) .* reshape(conj_kern_hat,[N 1 K C]);
x_hat = reshape(x_hat, [N, B, K*C]);

end
