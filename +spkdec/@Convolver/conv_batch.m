function y = conv_batch(self, x)
% Perform the forward convolution on short batches of data
%   y = conv_batch(self, x)
%
% Returns:
%   y       [T+L-1 x C x B] convolution output
% Required arguments:
%   x       [T x D x B] data to convolve with

% Some dimensions
[T, D, B] = size(x);
assert(D==self.D, self.errid_dim, 'x must be [T x D x B]');
C = self.C;
N = T + self.L - 1;
output_real = isreal(x) && isreal(self.kernels);

% Get the kernels in frequency domain
kern_hat = self.get_kernels_hat(N);

% Permute so we can reuse the old code
y = permute(x, [1 3 2]);        % [T x B x D]
% Perform the convolution in frequency domain
y = fft(y, N, 1);               % [N x B x D]
y = self.conv_hat(y, kern_hat); % [N x B x C]
if output_real, sym_flag='symmetric'; else, sym_flag = 'nonsymmetric'; end
y = ifft(y, N, 1, sym_flag);    % [N x B x C]
% Apply the cross-channel transform
y = reshape(y, [N*B,C]);        % [N*B x C]
y = y * self.wh_ch.';           % = (wh_ch * y.').'
y = reshape(y, [N B C]);        % [N x B x C]
% Permute back
y = permute(y, [1 3 2]);        % [N x C x B], where N = T+L-1

end
