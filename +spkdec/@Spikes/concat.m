function spk_all = concat(self, varargin)
% Concatenate a bunch of spikes together
%   spk_all = concat(spk_A, spk_B, ...)
%
% Returns:
%   spk_all     Spikes object containing all of the spikes
% Required arguments:
%   spk_A,...   Individual Spikes objects to concatenate together
%
% All of the spikes objects must have the same dimension of feature space.

% Convert the cell array into an object vector
spk_set = [self, varargin{:}];

% Check the dimensions
D = arrayfun(@(spk) size(spk.X,1), spk_set);
assert(all(D==D(1)), self.errid_dim, ...
    'All spike objects must have the same feature space dimension');

% Concatenate them
t = vertcat(spk_set.t);
r = vertcat(spk_set.r);
X = horzcat(spk_set.X);

% Sort if necessary
if ~issorted(t)
    [t, sortidx] = sort(t);
    r = r(sortidx);
    X = X(:,sortidx);
end

% Construct the output object
spk_all = spkdec.Spikes(t, r, X);

end
