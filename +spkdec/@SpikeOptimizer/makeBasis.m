function [basis_obj, spk] = makeBasis(self, spikes, K, varargin)
% Construct a new SpikeBasis optimized for the given spike waveforms
%   [basis, spk] = makeBasis(self, spikes, K, ...)
%
% Returns:
%   basis       New SpikeBasis object
%   spk         Spikes object (where t is the shift in detected spike time)
% Required arguments:
%   spikes      [L+W-1 x C x N] whitened spike waveforms
%   K           Number of spike basis waveforms per channel
% Optional parameters (key/value pairs) [default]:
%   t0          Sample index (1..L) for t=0                         [ 1 ]
%
% This initializes the basis waveforms using a singular value decomposition
% (SVD) on the given spike waveforms, then solves the optimization problem
% described in SpikeOptimizer.optimize().

% Parse optional inputs
ip = inputParser();
ip.addParameter('t0', 1, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Check some dimensions
[Lw, C, N] = size(spikes);
assert(Lw == self.L + self.W-1, self.errid_dim, ...
    'Given spike waveforms must have length L+W-1');
assert(C == self.C, self.errid_dim, 'Given spikes must have C channels');
assert(N > 2*K, self.errid_arg, 'The number of spikes must be >> K');

% Fit a spike basis to the given data
[basis, spk] = self.optimize(spikes, 'lambda',0, 'K',K);

% Flip the signs so that the average spike feature is positive
sgn = sign(mean(spk.X,2));
sgn = gather(double(sgn));
basis = basis .* reshape(sgn, [1 K C]);
spk.setFeat(sgn .* spk.X);

% Construct the SpikeBasis object
basis_obj = spkdec.SpikeBasis(basis, 't0',prm.t0, ...
    'whitener',self.whbasis.whitener, 'interp',self.whbasis.interp);

end
