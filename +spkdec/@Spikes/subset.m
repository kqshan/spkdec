function y = subset(self, mask)
% Return a subset of these spikes in a new Spikes object
%   y = subset(self, mask)
%
% Returns:
%   y       Selected subset of spikes (new Spikes object)
% Required arguments:
%   mask    [N x 1] logical mask or [m x 1] vector of spike indices

% Validate the input
mask = mask(:);
if islogical(mask)
    assert(length(mask)==self.N, self.errid_dim, ...
        'If mask is a logical vector, its length must match self.N');
elseif isnumeric(mask)
    assert(issorted(mask), self.errid_arg, ...
        'If mask is a vector of spike indices, it must be sorted');
else
    error(self.errid_arg, 'Unsupported datatype for <mask> argument');
end

% Construct a new Spikes object with this subset of spikes
if size(self.X,1)==0
    X = [];
else
    X = self.X(:,mask);
end
y = spkdec.Spikes(self.t(mask), self.r(mask), X);

end
