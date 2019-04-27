function overlaps = find_overlaps(t, r, dt_max)
% Find the overlapping spikes given a vector of spike times
%   overlaps = find_overlaps(t, r, dt_max)
%
% Returns:
%   overlaps    Struct with fields:
%     bands       [B x N] case index (1..P) for each overlap
%     cases       [P x 3] unique values of [lag, r1, r2]
% Required arguments:
%   t           [N x 1] spike time for each spike (sorted)
%   r           [N x 1] sub-sample shift index for each spike
%   dt_max      Maximum gap between spikes to consider
%
% The output struct tells us about the overlap between spike n and spike n+b-1:
%   p = overlaps.bands(b, n)
%   overlaps.cases(p,1) == t(n) - t(n+b-1)  % lag
%   overlaps.cases(p,2) == r(n+b-1)         % r1
%   overlaps.cases(p,3) == r(n)             % r2
%
% A few notes:
% * lag will be <= 0. This is a necessary consequence of how we defined the Gram
%   matrices and our decision to represent the lower diagonals.
% * Overlaps in which abs(lag) > dt_max will be assigned [lag=Inf, r1=1, r2=1].

% Dimensions
N = numel(t);

% Fill values for overlaps that are invalid or exceed the limit
lag_fill = Inf;
r_fill = 1;

% Evaluate lag, r1, and r2 row-by-row
% Start with a special case for zero lag
lag_rows = {zeros(1,N)};
r1_rows = {r'};
r2_rows = {r'};
% Build the matrix
for gap = 1:N-1
    % Define spike 1 and spike 2
    t1 = t(gap+1:end);  % t(n+b-1) : it may seem weird that this is spike 1, but
    t2 = t(1:end-gap);  % t(n)       that's how it works in the lower diagonal
    r1 = r(gap+1:end);
    r2 = r(1:end-gap);
    % Compute the lag and exit early if all exceed the limit
    lag = t2 - t1;
    exc_max = abs(lag) > dt_max;
    if all(exc_max)
        break
    end
    % Replace the max-exceeding lags with the fill value
    lag(exc_max) = lag_fill;
    r1(exc_max) = r_fill;
    r2(exc_max) = r_fill;
    % Append these data
    lag_rows{gap+1} = [lag; lag_fill*ones(gap,1)]';
    r1_rows{gap+1} = [r1; r_fill*ones(gap,1)]';
    r2_rows{gap+1} = [r2; r_fill*ones(gap,1)]';
end

% Concatenate the rows, building [B x N] matrices
lag = vertcat(lag_rows{:});
B = size(lag,1);
r1 = vertcat(r1_rows{:});
r2 = vertcat(r2_rows{:});

% Find the unique cases
% Reshape these into a [B*N x 3] combined matrix
lrr = [lag(:), r1(:), r2(:)];
% Find the unique cases
[cases, ~, case_indices] = unique(lrr, 'rows');
% Reshape the case indices back into a [B x N] matrix
case_indices = reshape(case_indices, [B N]);

% Create the output struct
overlaps = struct('bands',case_indices, 'cases',cases);

end
