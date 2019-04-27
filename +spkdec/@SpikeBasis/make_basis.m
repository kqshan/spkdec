function obj = make_basis(spikes, K, varargin)
% Initialize a SpikeBasis from detected spikes
%   obj = make_basis(spikes, K, ...)
%
% Returns:
%   obj         New SpikeBasis object
% Required arguments:
%   spikes      [L+W-1 x C x N] whitened spike waveforms
%   K           Number of spike basis waveforms per channel
% Optional parameters (key/value pairs) [default]:
%   t0          Sample index (1..L) for t=0             [ 1 ]
%   whitener    Whitener object                         [ none ]
%   interp      Interpolator object                     [ none ]
%   ...         Additional params forwarded to SpikeOptimizer constructor

% Inherit the error IDs
errid_arg = spkdec.SpikeBasis.errid_arg;
errid_dim = spkdec.SpikeBasis.errid_dim;

% Optional parameters
ip = inputParser();
ip.KeepUnmatched = true; ip.PartialMatching = false;
ip.addParameter('t0', 1, @isscalar);
ip.addParameter('whitener', [], @(x) isempty(x) || isa(x,'spkdec.Whitener'));
ip.addParameter('interp', [], @(x) isempty(x) || isa(x,'spkdec.Interpolator'));
ip.parse( varargin{:} );
prm = ip.Results;
extra_args = ip.Unmatched;

% Check some dimensions
[Lw, C, N] = size(spikes);
assert(N > 2*K, errid_arg, 'The number of spikes must be >> K');

% Get the whitener
whitener = prm.whitener;
if isempty(whitener)
    whitener = spkdec.Whitener.no_whiten(C);
else
    assert(whitener.C==C, errid_dim, 'whitener.C must match given spikes');
end
% Infer the data length
W = whitener.W;
L = Lw - (W-1);
assert(L >= 1, errid_dim, 'whitener.W is too long for the given spikes');

% Get the interpolator
interp = prm.interp;
if isempty(interp)
    interp = spkdec.Interpolator.no_interp(L);
else
    assert(interp.L==L, errid_dim, 'interp.L must match given spikes');
end

% Use a SpikeOptimizer to fit a spike basis to the given data
whbasis = spkdec.WhitenerBasis(whitener, 'interp',interp);
optimizer = spkdec.SpikeOptimizer(whbasis, extra_args);
[basis, spk] = optimizer.optimize(spikes, 'lambda',0, 'K',K, 'is_wh',true);

% Flip the signs of the basis so that average spike feature is positive
sgn = sign(mean(spk.X,2));
sgn = gather(double(sgn));
basis = basis .* reshape(sgn, [1 K C]);

% Construct the SpikeBasis object
obj = spkdec.SpikeBasis(basis, 't0',prm.t0, ...
    'whitener',whitener, 'interp',interp);

end
