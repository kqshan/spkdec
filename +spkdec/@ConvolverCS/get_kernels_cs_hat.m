function kcs_hat = get_kernels_cs_hat(self, N)
% Populate the cache with the kernel waveforms in DFT domain
%   kcs_hat = get_kernels_cs_hat(self, N);
%
% Returns:
%   kcs_hat [N x K x C] channel-specific kernels in DFT domain (causal)
% Required arguments:
%   N       DFT size
%
% If N > L, then extra zeros will be added to the end if the kernel, the same as
% the behavior of fft().

% Look in the cache
if size(self.kernels_cs_hat,1)==N
    kcs_hat = self.kernels_cs_hat;
    return
end

% Perform the FFT
kcs_hat = fft(self.kernels_cs, N, 1);

% Cache the result
self.kernels_hat = kcs_hat;

end
