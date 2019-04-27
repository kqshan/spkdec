function mat = getGram(self, lag, r1, r2, skip_cache)
% Return the Gram matrix at a specified lag
%   mat = getGram(self, [lag, r1, r2])
%
% Returns:
%   mat       [D x D] Gram matrix (see below for definition)
% Optional arguments [default]:
%   lag       Time lag (#samples) applied to kernel j       [ 0 ]
%   r1        Kernel family index (1..R) for kernel i       [ 1 ]       
%   r2        Kernel family index (1..R) for kernel j       [ 1 ]
%
% The arguments (lag,r1,r2) can also be provided as [N x 1] vectors, in which
% case the returned mat will be a [D x D x N] matrix.
%
% If lag >= 0, then
%   mat(i,j) = < kernels(lag+1:end,:,i,r1) , kernels(1:end-lag,:,j,r2) >
% and if lag < 0, then
%   self.getGram(lag, r1, r2) = self.getGram(-lag, r1, r2).'

% Secret 5th argument:
%   skip_cache    Do not insert results into the cache      [ false ]

% Dimensions
[L, C, D, ~] = size(self.kernels);

% Optional arguments
if nargin < 2, lag = 0; end
if nargin < 3, r1 = 1; end
if nargin < 4, r2 = 1; end
if nargin < 5, skip_cache = false; end

% Expand any scalars to match, and make them all column vectors
N = max([numel(lag), numel(r1), numel(r2)]);
if isscalar(lag), lag = repmat(lag, [N 1]); else, lag = lag(:); end
if isscalar(r1) ,  r1 = repmat( r1, [N 1]); else,  r1 = r1(:);  end
if isscalar(r2) ,  r2 = repmat( r2, [N 1]); else,  r2 = r2(:);  end

% Process each case independently
mat = zeros(D, D, N);
for n = 1:N
    % Transpose the request if lag < 0
    lag_n = lag(n);
    do_transpose = (lag_n < 0);
    if do_transpose
        lag_n = -lag_n;
        r1_n = r2(n);
        r2_n = r1(n);
    else
        r1_n = r1(n);
        r2_n = r2(n);
    end
    
    % Special case if the lag is large
    if lag_n >= L
        % mat(:,:,n) = 0
        continue;
    end
    
    % Look for a cached result
    G = self.cache_request(r1_n, r2_n, lag_n);
    
    % Compute the result if necessary
    if isempty(G)
        % Get the kernels
        A1 = self.kernels(lag_n+1:end, :, :, r1_n);
        A2 = self.kernels(1:end-lag_n, :, :, r2_n);
        % Compute their dot product
        G = reshape(A1, [(L-lag_n)*C, D])' * reshape(A2, [(L-lag_n)*C, D]);
        % Cache the new result
        if ~skip_cache
            self.cache_write(r1_n, r2_n, lag_n, G);
        end
    end
    
    % Transpose as required
    if do_transpose
        G = G.';
    end
    mat(:,:,n) = G;
end

end
