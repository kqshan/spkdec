function X = toFull(self, T, R)
% Convert this Spikes object into a [T x D x R] array
%   spikes = toFull(self, T, [R])
%
% Returns:
%   spikes      [T x D x R] array of spike features (see SpikeBasis.conv)
% Required arguments:
%   T           Desired data length (#samples)
% Optional arguments [default]
%   R           Subsample interpolation ratio               [ 1 ]
%
% See also: spkdec.SpikeBasis.conv

if nargin < 3, R = 1; end

% Error messages
errid_arg = self.errid_arg;
errid_dim = self.errid_dim;
errmsg_dim = 'Given dimensions are not compatible with this Spikes object (%s)';

% Some local variables for the data of interest
spk_X = self.X;
[D,N] = size(spk_X);
assert(N==self.N, errid_arg, 'self.X must be set before calling tofull()');
spk_t = self.t;
assert(all(spk_t >= 1), errid_arg, 'self.t must be >= 1 when calling toFull()');
spk_r = self.r;

% Check that the dimensions are compatible
if (N > 0)
    assert(T >= max(spk_t), errid_dim, errmsg_dim, 'need T >= max(self.t)');
    assert(R >= max(spk_r), errid_dim, errmsg_dim, 'need R >= max(self.r)');
end

% Construct the output matrix
X = zeros(D, R*T, 'like',spk_X);
rt = double(spk_r) + R*double(spk_t-1);
X(:,rt) = spk_X;           % [D x R*T]
% Reshape into the desired form
X = reshape(X, [D R T]);   % [D x R x T]
X = permute(X, [3 1 2]);   % [T x D x R]

end
