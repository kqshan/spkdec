function y = vec_to_batch(x, N, ovlp, dupe)
% Convert a vector into a set of batches for overlap-add or overlap-scrap
%   y = vec_to_batch(x, N, ovlp, dupe)
%
% Returns:
%   y       [N x #batch x C] data in batches
% Required arguments:
%   x       [T x C] input data (all trailing dims will be collapsed into C)
%   N       Desired batch size
%   ovlp    Overlap between batches
%   dupe    Duplicate the overlap (set dupe=true for overlap-scrap)
%
% Here are some examples to illustrate the effect of the <dupe> option:
%    x = [ 1     vec_to_batch(x,3,1,false)     vec_to_batch(x,3,1,true)
%          2        = [ 1  3  5                   = [ 1  3
%          3            2  4  0                       2  4
%          4            0  0  0 ]                     3  5 ]
%          5 ]
% So in addition to duplicating the overlap regions, setting dupe=true may also
% change the number of batches.

% Dimensions
[T,C] = size(x);
stride = N - ovlp;

% The number of batches depends on whether we can include the trailing overlap
% (if present) as part of the last batch or not
if dupe
    nBatch = ceil((T-ovlp)/stride);
else
    nBatch = ceil(T/stride);
end

% Start with all zeros
y = zeros(N,nBatch,C, 'like',x);
% Fill in the complete batches
T1 = stride*(nBatch-1);
y(1:stride, 1:nBatch-1, :) = reshape(x(1:T1,:), [stride, nBatch-1, C]);
% And the remaining batch
T_left = T - T1;
y(1:T_left, nBatch, :) = reshape(x(T1+1:T,:), [T_left, 1, C]);

% Duplicate the overlap if desired
if dupe
    y(stride+1:N, 1:nBatch-1, :) = y(1:ovlp, 2:nBatch, :);
end

end
