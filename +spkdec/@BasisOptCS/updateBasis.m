function [basis_obj, spk_new] = updateBasis(self, basis, spk, resid, varargin)
% Update a spike basis using proximal gradient descent
%   [basis_new, spk_new] = updateBasis(self, basis, spk, resid, ...)
%
% Returns:
%   basis_new   New SpikeBasisCS with updated basis waveforms
%   spk_new     Detected spikes (Spikes object) after optimization
% Required arguments:
%   basis       SpikeBasisCS object
%   spk         Detected spikes (Spikes object)
%   resid       [L+W-1 x C x N] spike residuals (whitened)
% Optional parameters (key/value pairs) [default]:
%   lambda      Regularizer param in optimize()             [defer to reg_wt]
%   reg_wt      Relative weight to put on regularizer       [ 0.1 ]
%   D_add       Number of basis waveforms to add            [ 0 ]
%   ...         Add'l parameters are forwarded to optimize()
%
% This calls self.optimize() to solve the following optimization problem:
%       minimize    ||Y - basis.reconst(X)||^2 + lambda*||basis-basis_old||^2
%     subject to    basis is channel-specific and channelwise orthonormal
% where the norms are defined in terms of the whitened inner product, and Y is
% the spike waveforms (Y = basis_old.reconst(spk_old) + resid).
%
% By default, this sets lambda = reg_wt * ||spikes||^2/||basis||^2, but this can
% be overridden by explicitly specifying the <lambda> parameter.
%
% If D_add > 0, then ihis initializes the new basis waveforms using a singular
% value decomposition (SVD) of the given residual.

% Make sure this is a channel-specific basis
assert(isa(basis,'spkdec.SpikeBasisCS'), self.errid_arg, ...
    'This channel-specific optimizer requires a SpikeBasisCS input argument');

% Make sure that D_add is divisible by C (this will be enforced later on, but
% the error message is less helpful because it refers to `D` instead of `D_add`)
ip = inputParser();
ip.KeepUnmatched = true; ip.PartialMatching = false;
ip.addParameter('D_add', 0, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;
assert(mod(prm.D_add, self.C)==0, self.errid_arg, ...
    'For this channel-specific optimizer, D_add must be divisible by C');

% Call the parent method
[basis_obj, spk_new] = updateBasis@spkdec.BasisOptimizer( ...
    self, basis, spk, resid, varargin{:} );

% Convert the basis object into a channel-specific basis
basis_obj = spkdec.SpikeBasisCS.from_basis(basis_obj);

end
