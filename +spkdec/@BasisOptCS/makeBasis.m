function [basis_obj, spk] = makeBasis(self, spikes, D, varargin)
% Construct a new SpikeBasisCS optimized for the given spike waveforms
%   [basis, spk] = makeBasis(self, spikes, D, ...)
%
% Returns:
%   basis       New SpikeBasisCS object
%   spk         Spikes object (where t is the shift in detected spike time)
% Required arguments:
%   spikes      [L+W-1 x C x N] whitened spike waveforms
%   D           Desired number of spike basis waveforms (must be divisible by C)
% Optional parameters (key/value pairs) [default]:
%   t0          Sample index (1..L) for t=0                         [ 1 ]
%   ...         Add'l params are forwarded to self.optimize()
%
% This initializes the basis waveforms using a singular value decomposition
% (SVD) on the given spike waveforms, then solves the optimization problem
% described in optimize().

% Call the parent method
[basis_obj,spk] = makeBasis@spkdec.BasisOptimizer(self, spikes, D, varargin{:});

% Convert the basis object into a channel-specific basis
basis_obj = spkdec.SpikeBasisCS.from_basis(basis_obj);

end
