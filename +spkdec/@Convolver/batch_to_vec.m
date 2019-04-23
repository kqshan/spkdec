function x = batch_to_vec(y, T, ovlp, add)
% Convert a set of overlap-add or overlap-scrap batches into a vector
%   x = batch_to_vec(y, T, ovlp, add)
%
% Returns:
%   x       [T x C] data, with overlap either added or scrapped
% Required arguments:
%   y       [N x #batch x C] input data (trailing dims will be collapsed into C)
%   T       Desired output length
%   ovlp    Overlap between batches
%   add     Add the overlap when constructing the output

% Dimensions
[N, nBatch, C] = size(y);
stride = N - ovlp;

% Extract a stride's worth of data from each batch
x = y(1:stride, :, :);
% Add the overlap if desired
if add
    x(1:ovlp,2:nBatch,:) = x(1:ovlp,2:nBatch,:) + y(stride+1:N,1:nBatch-1,:);
end
% Concatenate
T1 = stride * nBatch;
x = reshape(x, [T1, C]);

% Truncate or append the final overlap
if T <= T1
    x = x(1:T, :);
else
    assert(add, 'Having T > (N-ovlp)*nBatch is only valid when add==true');
    T_extra = T - T1;
    x_extra = y(stride+(1:T_extra), nBatch, :);
    x = [x; reshape(x_extra, [T_extra, C])];
end

end
