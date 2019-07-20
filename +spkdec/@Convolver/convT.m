function x = convT(self, y, varargin)
% Perform the transpose convolution
%   x = convT(self, y, varargin)
%
% Returns:
%   x       [T-L+1 x D] convolution output
% Required arguments:
%   y       [T x C] input data
% Optional parameters (key/value pairs) [default]:
%   N_fft   FFT size for overlap-scrap              [ auto ]

% Some dimensions
[T, C] = size(y);
assert(C==self.C, self.errid_dim, 'y must be [T x C]');
ovlp = self.L - 1;
N_auto = max(1024, pow2(2+nextpow2(ovlp+1)));
output_real = isreal(y) && isreal(self.kernels);

% Optional parameters
ip = inputParser();
ip.addParameter('N_fft', N_auto, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Get the kernels in frequency domain
N = prm.N_fft;
assert(N >= 2 * self.L, self.errid_arg, ...
    'N_fft must be >= 2*self.L to ensure correct results from overlap-add');

% Perform the convolution using overlap-scrap
%   Create the batches
x = self.vec_to_batch(y, N, ovlp, true);
%   Perform the convolution in frequency domain
x = fft(x, N, 1);
x = self.convT_hat(x);
if output_real, sym_flag='symmetric'; else, sym_flag = 'nonsymmetric'; end
x = ifft(x, N, 1, sym_flag);
%   Overlap-scrap and convert back into a vector
x = self.batch_to_vec(x, T-ovlp, ovlp, false);

end
