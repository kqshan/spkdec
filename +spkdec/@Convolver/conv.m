function y = conv(self, x, varargin)
% Perform the forward convolution
%   y = conv(self, x, ...)
%
% Returns:
%   y       [T+L-1 x C] convolution output
% Required arguments:
%   x       [T x D] data to convolve with
% Optional parameters (key/value pairs) [default]:
%   N_fft   FFT size for overlap-add                [ auto ]

% Some dimensions
[T, D] = size(x);
assert(D==self.D, self.errid_dim, 'x must be [T x D]');
ovlp = self.L - 1;
N_auto = max(1024, pow2(2+nextpow2(ovlp+1)));
output_real = isreal(x) && isreal(self.kernels);

% Optional parameters
ip = inputParser();
ip.addParameter('N_fft', N_auto, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Get the kernels in frequency domain
N = prm.N_fft;
assert(N >= 2 * self.L, self.errid_arg, ...
    'N_fft must be >= 2*self.L to ensure correct results from overlap-add');
kern_hat = self.get_kernels_hat(N);

% Perform the convolution using overlap-add
%   Create the batches
y = self.vec_to_batch(x, N, ovlp, false);
%   Perform the convolution in frequency domain
y = fft(y, N, 1);
y = self.conv_hat(y, kern_hat);
if output_real, sym_flag='symmetric'; else, sym_flag = 'nonsymmetric'; end
y = ifft(y, N, 1, sym_flag);
%   Overlap-add and convert back into a vector
y = self.batch_to_vec(y, T+ovlp, ovlp, true);

% Apply the cross-channel transform
y = y * self.wh_ch.'; % = (wh_ch * y.').'

end
