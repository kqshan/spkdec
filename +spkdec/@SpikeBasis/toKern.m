function kern = toKern(self, varargin)
% Return a matrix representing the whitened basis waveforms
%   kern = toKern(self, varargin)
%
% Returns:
%   kern        [L+W-1 x C x K*R*C] matrix representing the whitened basis
%               waveforms as a linear map: [K*R*C] --> [L+W-1 x C]
% Optional parameters (key/value pairs) [default]:
%   flatten     Flatten output into a [(L+W-1)*C x K*R*C] matrix      [ false ]
%
% This produces the same result as self.toConv().toKern()

ip = inputParser();
ip.addParameter('flatten', false, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Dimensions
[L, K, C] = size(self.basis);
W = self.whitener.W;
R = self.interp.R;

% Represent the whitener as an explicit matrix
whitener_mat = self.whitener.toMat(L);
Lw = L + W-1;
whitener_mat = reshape(whitener_mat,[Lw*C, L, C]);  % [Lw*C x L x C]

% Whiten the kernels
kern = zeros(Lw*C, K, R, C);
for c = 1:C
    wh_mat = whitener_mat(:,:,c);
    for r = 1:R
        kern(:,:,r,c) = wh_mat * self.interp.shifts(:,:,r) * self.basis(:,:,c);
    end
end

% Reshape as desired
if prm.flatten
    kern = reshape(kern, [Lw*C, K*R*C]);
else
    kern = reshape(kern, [Lw, C, K*R*C]);
end

end
