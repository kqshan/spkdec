function y_hat = conv_hat(x_hat, kern_hat)
% Perform the convolution (and summing over kernels) in frequency domain
%   y_hat = conv_hat(x_hat, kern_hat)
%
% Returns:
%   y_hat       [N x B x C] convolution-and-sum output (in freq. domain)
% Required arguments:
%   x_hat       [N x B x D] input data in frequency domain
%   kern_hat    [N x K x C] convolution kernels in frequency domain (D = K*C)

% Dimensions
[N, B, D] = size(x_hat);
[N_, K, C] = size(kern_hat);
assert(N==N_ && K*C==D);

% Multiply
y_hat = reshape(x_hat,[N B K C]) .* reshape(kern_hat, [N 1 K C]);
% Sum
y_hat = sum(y_hat, 3);
y_hat = reshape(y_hat, [N B C]);

end
