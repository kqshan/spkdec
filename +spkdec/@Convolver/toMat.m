function A = toMat(self, varargin)
% Return a matrix representing the effective kernels of this object
%   mat = toMat(self, ...)
%
% Returns:
%   mat         [L x C x K x C] matrix representing these kernels (including the
%               cross-channel transform) as a linear map: [K x C] --> [L x C]
% Optional parameters (key/value pairs) [default]:
%   flatten     Flatten output into a [L*C x K*C] matrix        [ false ]

% Optional parameters
ip = inputParser();
ip.addParameter('flatten', false, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Use the cache if available
if isempty(self.kernels_full)
    % Block-diagonalize our kernels
    A = self.kernels;                   % [L x K x C]
    [L,K,C] = size(A);
    A = mat2cell(A, L, K, ones(C,1));   % {C x 1} [L x K]
    A = blkdiag(A{:});                  % [L*C x K*C]
    A = reshape(A, [L C K C]);          % [L x C x K x C]
    % Apply the cross-channel trnasform
    for kc = 1:K*C
        A(:,:,kc) = A(:,:,kc) * self.wh_ch.'; % = (wh_ch * A(:,:,kc).').'
    end
    % Cache the result
    self.kernels_full = A;
else
    % Use the cached result
    A = self.kernels_full;
end

% Reshape if desired
if prm.flatten
    [L,C,K,~] = size(A);
    A = reshape(A, [L*C, K*C]);
end

end
