function A = toKern(self, varargin)
% Return a matrix representing the effective kernels of this object
%   mat = toKern(self, ...)
%
% Returns:
%   mat         [L x C x D] matrix representing these kernels (including the
%               cross-channel transform) as a linear map: [D] --> [L x C]
% Optional parameters (key/value pairs) [default]:
%   flatten     Flatten output into a [L*C x D] matrix          [ false ]
%
% This differs from Whitener.toMat in that this method returns the kernels for a
% single time step, whereas Whitener.toMat produces a Toeplitz matrix that 
% represents the whitening operator for a particular input duration.

% Optional parameters
ip = inputParser();
ip.addParameter('flatten', false, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Use the cache if available
if isempty(self.kernels_full)
    % Block-diagonalize our kernels
    A = self.kernels;                   % [L x K x C]
    [L,K,C] = size(A); D = K*C;
    A = mat2cell(A, L, K, ones(C,1));   % {C x 1} [L x K]
    A = blkdiag(A{:});                  % [L*C x K*C]
    A = reshape(A, [L C D]);            % [L x C x D] (since D = K*C)
    % Apply the cross-channel trnasform
    for d = 1:D
        A(:,:,d) = A(:,:,d) * self.wh_ch.'; % = (wh_ch * A(:,:,kc).').'
    end
    % Cache the result
    self.kernels_full = A;
else
    % Use the cached result
    A = self.kernels_full;
end

% Reshape if desired
if prm.flatten
    [L,C,D] = size(A);
    A = reshape(A, [L*C, D]);
end

end
