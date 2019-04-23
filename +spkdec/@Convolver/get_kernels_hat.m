function k_hat = get_kernels_hat(self, N)
% Populate the cache with the kernel waveforms in DFT domain
%   k_hat = get_kernels_hat(self, N)
%
% Returns:
%   k_hat   [N x K x C] single-channel kernels in DFT domain (as causal kernels)
% Required arguments:
%   N       DFT size

% Look in the cache
if size(self.kernels_hat,1)==N
    k_hat = self.kernels_hat;
    return
end

% Check the dimension
assert(N >= 2 * self.L, self.errid_arg, ...
    'N_fft must be >= 2*self.L to ensure correct results from overlap-add');

% Perform the FFT
k_hat = fft(self.kernels, N, 1);

% Cache the result
self.kernels_hat = k_hat;

end
