function kernels = compute_kernels(kernels_cs, wh_ch)
% Compute the full kernels from the channel-specific ones
%   kernels = compute_kernels(kernels_cs, wh_ch);
%
% Returns:
%   kernels     [L x C x D] kernels (where D = K*C)
% Required arguments:
%   kernels_cs  [L x K x C] channel-speciifc kernels
%   wh_ch       [C x C] cross-channel transform

% Block-diagonalize our kernels
A = kernels_cs;                     % [L x K x C]
[L,K,C] = size(A); D = K*C;
A = mat2cell(A, L, K, ones(C,1));   % {C x 1} [L x K]
A = blkdiag(A{:});                  % [L*C x K*C]
A = reshape(A, [L C D]);            % [L x C x D] (since D = K*C)

% Apply the cross-channel transform
for d = 1:D
    A(:,:,d) = A(:,:,d) * wh_ch.';  % = (wh_ch * A(:,:,kc).').'
end

% Return the result
kernels = A;

end
