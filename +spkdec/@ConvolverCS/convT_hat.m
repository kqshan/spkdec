function x_hat = convT_hat(self, y_hat)
% Perform the transpose convolution (and sum over channels) in frequency domain
%   x_hat = convT_hat(self, y_hat)
%
% Returns:
%   x_hat       [N x B x D] convolution output (in frequency domain) (D = K*C)
% Required arguments:
%   y_hat       [N x B x C] input data in frequency domain
%
% The D = K*C columns of x_hat will be ordered by K first, then C.

% Get the [N x K x C] channel-specific convolution kernels in freq domain
[N, B, C] = size(y_hat);
kcs_hat = self.get_kernels_cs_hat(N);
K = size(kcs_hat,2);

% Apply the cross-channel transform
x_hat = reshape(y_hat, [N*B, C]);   % [N*B x C]
x_hat = x_hat * conj(self.wh_ch);   % = (wh_ch' * y.').'

% Multiply by the conjugate
x_hat = reshape(x_hat, [N B 1 C]) .* reshape(conj(kcs_hat),[N 1 K C]);
x_hat = reshape(x_hat, [N, B, K*C]);

end
