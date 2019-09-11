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
%   basis_mode  Basis mode: {'channel-specific',['omni-channel']}
%
% This initializes the basis waveforms using a singular value decomposition
% (SVD) on the given spike waveforms, then solves the optimization problem
% described in BasisOptimizer.optimize().

% Parse optional inputs
ip = inputParser();
ip.addParameter('t0', 1, @isscalar);
ip.addParameter('basis_mode', 'omni-channel', @ischar);
ip.parse( varargin{:} );
prm = ip.Results;

% Check some dimensions
[Lw, C, N] = size(spikes);
assert(Lw == self.L + self.W-1, self.errid_dim, ...
    'Given spike waveforms must have length L+W-1');
assert(C == self.C, self.errid_dim, 'Given spikes must have C channels');
% Some of these depend on the basis mode
basis_mode = prm.basis_mode;
switch basis_mode
    case 'channel-specific'
        assert(mod(D,C)==0, self.errid_arg, ...
            'In the "channel-specific" basis mode, D must be divisible by C');
        min_N = D/C;
    case 'omni-channel'
        min_N = D;
    otherwise
        error(self.errid_arg, 'Unsupported mode "%s"',basis_mode);
end
assert(N > 2*min_N, self.errid_arg, 'The number of spikes must be >> D');

% Fit a spike basis to the given data
switch basis_mode
    case 'channel-specific'
        [basis_cs, spk] = self.optimizeCS(spikes, 'lambda',0, 'K',D/C);
    case 'omni-channel'
        [basis, spk] = self.optimize(spikes, 'lambda',0, 'D',D);
end

% Flip the signs so that the average spike feature is positive
sgn = sign(mean(spk.X,2));
sgn = gather(double(sgn));
spk.setFeat(sgn .* spk.X);
% Still need to flip the sign of the basis waveforms

% Construct the SpikeBasis object
basis_prm = struct('t0',prm.t0, 'whitener',self.whbasis.whitener, ...
    'interp',self.whbasis.interp);
switch basis_mode
    case 'channel-specific'
        basis_cs = basis_cs .* reshape(sgn, [1, D/C, C]);
        basis_obj = spkdec.SpikeBasisCS(basis_cs, basis_prm);
    case 'omni-channel'
        basis = basis .* shiftdim(sgn,-2);
        basis_obj = spkdec.SpikeBasis(basis, basis_prm);
end

end
