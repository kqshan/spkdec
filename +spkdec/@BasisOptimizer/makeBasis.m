function [basis_obj, spk] = makeBasis(self, spikes, D, varargin)
% Construct a new SpikeBasis optimized for the given spike waveforms
%   [basis, spk] = makeBasis(self, spikes, D, ...)
%
% Returns:
%   basis       New SpikeBasis object
%   spk         Spikes object (where t is the shift in detected spike time)
% Required arguments:
%   spikes      [L+W-1 x C x N] whitened spike waveforms
%   D           Desired number of spike basis waveforms
% Optional parameters (key/value pairs) [default]:
%   t0          Sample index (1..L) for t=0                         [ 1 ]
%   ...         Add'l params are forwarded to self.optimize()
%
% This initializes the basis waveforms using a singular value decomposition
% (SVD) on the given spike waveforms, then solves the optimization problem
% described in BasisOptimizer.optimize().

% Parse optional inputs
ip = inputParser();
ip.KeepUnmatched = true; ip.PartialMatching = false;
ip.addParameter('t0', 1, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;
addl_args = ip.Unmatched;

% Fit a spike basis to the given data
[basis, spk] = self.optimize(spikes, 'lambda',0, 'D',D, addl_args);

% Flip the signs so that the average spike feature is positive
sgn = sign(mean(spk.X,2));
sgn = gather(double(sgn));
spk.setFeat(sgn .* spk.X);
basis = basis .* shiftdim(sgn,-2);

% Construct the SpikeBasis object
basis_obj = spkdec.SpikeBasis(basis, 'whitener',self.whbasis.whitener, ...
    't0',prm.t0, 'interp',self.whbasis.interp);

end
