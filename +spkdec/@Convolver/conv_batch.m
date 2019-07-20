function y = conv_batch(self, x)
% Perform the forward convolution on short batches of data
%   y = conv_batch(self, x)
%
% Returns:
%   y       [T+L-1 x C x B] convolution output
% Required arguments:
%   x       [T x D x B] data to convolve with

% Some dimensions
[T, D, B] = size(x); %#ok<ASGLU>
assert(D==self.D, self.errid_dim, 'x must be [T x D x B]');
N = T + self.L - 1;
output_real = isreal(x) && isreal(self.kernels);

% Permute so we can reuse the old code
y = permute(x, [1 3 2]);        % [T x B x D]
% Perform the convolution in frequency domain
y = fft(y, N, 1);               % [N x B x D]
y = self.conv_hat(y);           % [N x B x C]
if output_real, sym_flag='symmetric'; else, sym_flag = 'nonsymmetric'; end
y = ifft(y, N, 1, sym_flag);    % [N x B x C]
% Permute back
y = permute(y, [1 3 2]);        % [N x C x B], where N = T+L-1

end
