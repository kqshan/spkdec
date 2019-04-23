function x = convT(self, y, varargin)
% Perform the transpose convolution
%   x = convT(self, y, varargin)
%
% Returns:
%   x       [T-L+1 x K x C] convolution output
% Required arguments:
%   y       [T x C] input data
% Optional parametes (key/value pairs) [default]:
%   N_fft   FFT size for overlap-scrap              [ auto ]

% Some dimensions
[T, C] = size(y);
assert(C==self.C, self.errid_dim, 'y must be [T x C]');
K = self.K;
ovlp = self.L - 1;
N_auto = max(1024, pow2(2+nextpow2(ovlp+1)));

% Optional parameters
ip = inputParser();
ip.addParameter('N_fft', N_auto, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Get the kernels in frequency domain
N = prm.N_fft;
kern_hat = self.get_kernels_hat(N);

% Apply the cross-channel transform
y = y * conj(self.wh_ch); % = (wh_ch' * y.').'

% Perform the convolution using overlap-scrap
%   Create the batches
x = self.vec_to_batch(y, N, ovlp, true);
%   Perform the convolution in frequency domain
x = fft(x, N, 1);
x = self.convT_hat(x, kern_hat);
x = ifft(x, N, 1);
%   Enforce realness
if isreal(y) && isreal(self.kernels), x = real(x); end
%   Overlap-scrap and convert back into a vector
x = self.batch_to_vec(x, T-ovlp, ovlp, false);
x = reshape(x, [T-ovlp, K, C]);

end
