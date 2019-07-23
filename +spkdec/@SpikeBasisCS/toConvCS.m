function conv = toConvCS(self)
% Return a spkdec.ConvolverCS object for the whitened basis waveforms
%   conv = toConvCS(self)
%
% This ConvolverCS object has K*R kernels per channel, so conv.kernels_cs should
% be interpreted as a reshaped version of a [L x K x R x C] array.

% Use the cache if available
conv = self.convolver_cs;
if ~isempty(conv), return; end

% Dimensions
[L, K, C] = size(self.basis_cs);
whitener = self.whitener;
W = whitener.W;
R = self.interp.R;

% Start by applying the interpolation and padding the filter delay
Lw = L + W-1;
kern = zeros(Lw, K, R, C);                  % [Lw x K x R x C]
for r = 1:R
    shift = self.interp.shifts(:,:,r);      % [L x L]
    for c = 1:C
        kern(1:L, :, r, c) = shift * self.basis_cs(:,:,c);
    end
end

% Apply the whitening filter to the basis waveforms (note that this is different
% from the output of self.toKern(), which also includes cross-channel whitening)
kern = reshape(kern, [Lw, K*R, C]);         % [Lw x K*R x C]
for c = 1:C
    kern(:,:,c) = filter(whitener.wh_filt(:,c), 1, kern(:,:,c));
end

% Construct the Convolver object
wh_t0 = self.t0 + whitener.delay;
conv = spkdec.ConvolverCS(kern, 'wh_ch',whitener.wh_ch, 't0',wh_t0);

% Cache this result
self.convolver_cs = conv;

end
