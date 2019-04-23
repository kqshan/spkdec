function y_hat = conv_hat(x_hat, kern_hat)
% Perform the convolution (and summing over kernels) in frequency domain
%   y_hat = conv_hat(x_hat, kern_hat)
%
% Returns:
%   y_hat       [N x B x C] convolution-and-sum output (in freq. domain)
% Required arguments:
%   x_hat       [N x B x K x C] input data in frequency domain
%   kern_hat    [N x K x C] convolution kernels in frequency domain

% Dimensions
[N, B, K, C] = size(x_hat);

% Multiply
y_hat = x_hat .* reshape(kern_hat, [N 1 K C]);
% Sum
y_hat = sum(y_hat, 3);
y_hat = reshape(y_hat, [N B C]);

end
