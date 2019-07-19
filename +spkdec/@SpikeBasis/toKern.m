function kern = toKern(self, varargin)
% Return a matrix representing the whitened basis waveforms
%   kern = toKern(self, varargin)
%
% Returns:
%   kern        [L+W-1 x C x D x R] matrix representing the whitened basis
%               waveforms as a linear map: [D x R] --> [L+W-1 x C]
% Optional parameters (key/value pairs) [default]:
%   flatten     Flatten output into a [(L+W-1)*C x D*R] matrix      [ false ]
%
% Note that this does not produce the same result as self.toConv().toKern(),
% which uses a different permutation of the waveform order (one that is probably
% less useful for an end user, but convenient for the Convolver object).

ip = inputParser();
ip.addParameter('flatten', false, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Dimensions
[L, K, C] = size(self.basis); D = K*C;
W = self.whitener.W;
R = self.interp.R;

% Represent the whitener as an explicit matrix
whitener_mat = self.whitener.toMat(L);
Lw = L + W-1;
whitener_mat = reshape(whitener_mat,[Lw*C, L, C]);  % [Lw*C x L x C]

% Whiten the kernels
kern = zeros(Lw*C, K, C, R);
for c = 1:C
    wh_mat = whitener_mat(:,:,c);
    for r = 1:R
        kern(:,:,c,r) = wh_mat * self.interp.shifts(:,:,r) * self.basis(:,:,c);
    end
end

% Reshape as desired
if prm.flatten
    kern = reshape(kern, [Lw*C, D*R]);
else
    kern = reshape(kern, [Lw, C, D, R]);
end

end
