function mask = is_reg_max(t, x, r)
% Determine which of the selected peaks are regional maxima
%   mask = is_reg_max(t, x, r)
%
% Returns:
%   mask    [N x 1] logical indicating which peaks are regional maxima
% Required arguments
%   t       [N x 1] indices (1..T) of peaks (i.e. local maxima)
%   x       [T x 1] source data
%   r       Radius defining the size of the region
%
% If t includes all the local maxima, then this guarantees that for every 
% tt in t(mask), we have x(tt) >= x(tt-r:tt+r).
%
% If there are distinct points tied for regional max (which is unlikely but does
% happen, especially with single-precision data), then the earlier point will be
% selected as a regional maximum. This ensures that diff(t(mask)) > r.

% Dimensions
assert(isvector(x), 'spkdec:Math:is_reg_max:BadArg', 'x must be a vector');
T = length(x);
N = length(t);

% Special case if r == 1
if (r==1)
    mask = true(N,1);
    return
end

% There's some weirdness with GPU arrayfun
t = gather(t);
x_t = gather(x(t));

% For each peak, find the largest peak within +/- r samples of it
% p1 = first index occurring on or after t-r
[~,last_before_radius] = histc(t - (r+0.5), t);
p1 = last_before_radius + 1;
% p2 = last index occurring on or before t+r
[~,p2] = histc(t + r, [t; Inf]);
% Find the largest peak in each of these ranges
[reg_max, reg_max_idx] = arrayfun(@(a,b) max(x_t(a:b)), p1, p2);
mask = (p1 + reg_max_idx-1 == (1:N)');

% Also check the boundaries
x_t1 = gather(x(max(1,t-r)));
x_t2 = gather(x(min(T,t+r)));
mask = mask & (reg_max > x_t1) & (reg_max > x_t2);

end
