function basis_obj = makeBasis(self, spikes, K, varargin)
% Construct a new SpikeBasis optimized for the given spike waveforms
%   basis = makeBasis(self, spikes, K, ...)
%
% Returns:
%   basis       New SpikeBasis object
% Required arguments:
%   spikes      [L+W-1 x C x N] whitened spike waveforms
%   K           Number of spike basis waveforms per channel
% Optional parameters (key/value pairs) [default]:
%   t0          Sample index (1..L) for t=0                         [ 1 ]
%   flip_signs  Flip the sign so the mean spike feature is positive [ true ]

% Parse optional inputs
ip = inputParser();
ip.addParameter('t0', 1, @isscalar);
ip.addParameter('flip_signs', true, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Check some dimensions
[Lw, C, N] = size(spikes);
assert(Lw == self.L + self.W-1, self.errid_dim, ...
    'Given spike waveforms must have length L+W-1');
assert(C == self.C, self.errid_dim, 'Given spikes must have C channels');
assert(N > 2*K, self.errid_arg, 'The number of spikes must be >> K');

% Fit a spike basis to the given data
[basis, spk] = self.optimize(spikes, 'lambda',0, 'K',K, 'is_wh',true);

% Flip the signs so that the average spike feature is positive
if prm.flip_signs
    sgn = sign(mean(spk.X,2));
    sgn = gather(double(sgn));
    basis = basis .* reshape(sgn, [1 K C]);
end

% Construct the SpikeBasis object
basis_obj = spkdec.SpikeBasis(basis, 't0',prm.t0, ...
    'whitener',self.whbasis.whitener, 'interp',self.whbasis.interp);

end
