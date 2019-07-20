function y_hat = conv_hat(self, x_hat)
% Perform the convolution (and sum over kernels) in frequency domain
%   y_hat = conv_hat(self, x_hat)
%
% Returns:
%   y_hat       [N x B x C] convolution-and-sum output (in freq. domain)
% Required arguments:
%   x_hat       [N x B x D] input data in frequency domain
%
% This assumes that the D = K*C column of x_hat are ordered by K first, then C.

% Get the [N x K x C] channel-specific convolution kernels in freq domain
[N, B, D] = size(x_hat); %#ok<ASGLU>
kcs_hat = self.get_kernels_cs_hat(N);
[~, K, C] = size(kcs_hat);

% Multiply
y_hat = reshape(x_hat, [N B K C]) .* reshape(kcs_hat, [N 1 K C]);
% Sum over kernels
y_hat = sum(y_hat, 3);              % [N x B x 1 x C]

% Apply the cross-channel transform
y_hat = reshape(y_hat, [N*B, C]);   % [N*B x C]
y_hat = y_hat * self.wh_ch.';       % = (wh_ch * y.').'
y_hat = reshape(y_hat, [N B C]);    % [N x B x C]

end
