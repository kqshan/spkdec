function kern = toKern(self, varargin)
% Return a matrix representing the whitened basis waveforms
%   kern = toKern(self, varargin)
%
% Returns:
%   kern        [L+W-1 x C x D x R] matrix representing the whitened basis
%               waveforms as a linear map: [D x R] --> [L+W-1 x C]
% Optional parameters (key/value pairs) [default]:
%   flatten     Flatten output into a [(L+W-1)*C x D*R] matrix      [ false ]

ip = inputParser();
ip.addParameter('flatten', false, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Dimensions
[L, C, D] = size(self.basis);
W = self.whitener.W;
R = self.interp.R;

% Apply the sub-sample shifts
kern = zeros(L, C*D, R);                                % [L x C*D x R]
for r = 1:R
    kern(:,:,r) = self.interp.shifts(:,:,r) * self.basis(:,:);
end

% Whiten the kernels
whitener_mat = self.whitener.toMat(L, 'flatten',true);  % [Lw*C x L*C]
kern = reshape(kern, [L*C, D*R]);                       % [L*C x D*R]
kern = whitener_mat * kern;                             % [Lw*C x D*R]

% Reshape as desired
if ~prm.flatten
    kern = reshape(kern, [L+W-1, C, D, R]);
end

end
