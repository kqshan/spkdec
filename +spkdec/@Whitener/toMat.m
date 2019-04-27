function A = toMat(self, L, varargin)
% Return a matrix implementing this whitener for a finite support
%   mat = toMat(self, L, ...)
%
% Returns:
%   mat         [L+W-1 x C x L x C] matrix representing this whitener as a
%               linear map: [L x C] --> [L+W-1 x C] (Toeplitz over the L dim)
% Required arguments:
%   L           Support duration (#samples)
% Optional parameters (key/value pairs) [default]:
%   flatten     Flatten output into a [(L+W-1)*C x L*C] matrix      [ false ]

% Optional parameters
ip = inputParser();
ip.addParameter('flatten', false, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Dimensions
W = self.W;
C = self.C;
L_wh = L + W - 1;
if prm.flatten
    output_dims = [L_wh*C, L*C];
else
    output_dims = [L_wh, C, L, C];
end

% Use the cache if available
if size(self.whitener_mat,1) == L_wh
    A = self.whitener_mat;
    A = reshape(A, output_dims);
    return
end

% Construct the [L_wh x L] frequency-whitening operator for each channel
wh_filt = self.wh_filt;
A_ch = zeros(L_wh, L, C);
for c = 1:C
    wf = wh_filt(:,c);                    % [W x 1]
    wf = [repmat(wf,[1 L]); zeros(L,L)];  % [W+L x L]
    wf = reshape(wf(1:end-L), [L_wh, L]); % [W+L-1 x L]
    A_ch(:,:,c) = wf;
end

% Block-diagonalize this into a [L_wh*C x L*C] matrix
A = mat2cell(A_ch, L_wh, L, ones(C,1)); % {C x 1} [L_wh x L]
A = blkdiag(A{:});                      % [L_wh*C x L*C]

% Apply the cross-channel whitening
A = reshape(A, [L_wh, C, L*C]);
for lc = 1:L*C
    A(:,:,lc) = A(:,:,lc) * self.wh_ch.'; % = (wh_ch * A(:,:,lc).').'
end

% Cache the result
self.whitener_mat = A;

% Reshape into the final dimensions
A = reshape(A, output_dims);

end
