function k_hat = get_kernels_hat(self, N)
% Populate the cache with the kernel waveforms in DFT domain
%   k_hat = get_kernels_hat(self, N)
%
% Returns:
%   k_hat   [N x K x C] single-channel kernels in DFT domain (as causal kernels)
% Required arguments:
%   N       DFT size
%
% If N > L, then extra zeros will be added to the end if the kernel, the same as
% the behavior of fft().

% Look in the cache
if size(self.kernels_hat,1)==N
    k_hat = self.kernels_hat;
    return
end

% Perform the FFT
k_hat = fft(self.kernels, N, 1);

% Cache the result
self.kernels_hat = k_hat;

end
