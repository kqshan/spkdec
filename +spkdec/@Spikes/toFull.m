function X = toFull(self, varargin)
% Convert this Spikes object into a [T x K x R x C] array
%   spikes = toFull(self, dims_vec)
%   spikes = toFull(self, T, dims_obj)
%
% Returns:
%   spikes      [T x K x R x C] array of spike features (see SpikeBasis.conv)
% Required arguments:
%   dims_vec    Desired dimensions as a 4-element vector ([T K R C])
% ---- or ----
%   T           Desired data length (#samples)
%   dims_obj    Struct/object with fields/properties defining K, R, C
%
% See also: spkdec.SpikeBasis.conv

% Error messages
errid_arg = self.errid_arg;
errid_dim = self.errid_dim;
errmsg_dim = 'Given dimensions are not compatible with this Spikes object (%s)';

% Obtain the dimensions
switch nargin
    case 2
        dims = varargin{1};
        assert(numel(dims)==4, errid_arg, ...
            'dims_vec must be a 4-element vector ([T K R C])');
        T = dims(1); K = dims(2); R = dims(3); C = dims(4);
    case 3
        [T, dims] = deal(varargin{:});
        K = dims.K; R = dims.R; C = dims.C;
    otherwise
        error(errid_arg, 'Incorrect number of input arguments');
end

% Some local variables for the data of interest
spk_X = self.X;
[D,N] = size(spk_X);
assert(N==self.N, errid_arg, 'self.X must be set before calling tofull()');
spk_t = self.t;
assert(all(spk_t >= 1), errid_arg, 'self.t must be >= 1 when calling toFull()');
spk_r = self.r;

% Check that the dimensions are compatible
assert(K*C == D, errid_dim, errmsg_dim, 'need K*C == self.D');
assert(T >= max(spk_t), errid_dim, errmsg_dim, 'need T >= max(self.t)');
assert(R >= max(spk_r), errid_dim, errmsg_dim, 'need R >= max(self.r)');

% Construct the output matrix
X = zeros(K*C, R*T, 'like',spk_X);
rt = double(spk_r) + R*double(spk_t-1);
X(:,rt) = spk_X;           % [K*C x R*T]
% Reshape into the desired form
X = reshape(X, [K C R T]); % [K x C x R x T]
X = permute(X, [4 1 3 2]); % [T x K x R x C]

end
