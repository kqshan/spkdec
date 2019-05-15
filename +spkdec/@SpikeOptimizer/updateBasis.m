function [basis_obj, spk_new] = updateBasis(self, basis, spk, resid, varargin)
% Update a spike basis using proximal gradient descent
%   [basis_new, spk_new] = updateBasis(self, basis, spk, resid, ...)
%
% Returns:
%   basis_new   New SpikeBasis with updated basis waveforms
%   spk_new     Detected spikes (Spikes object) after optimization
% Required arguments:
%   basis       SpikeBasis object
%   spk         Detected spikes (Spikes object)
%   resid       [L+W-1 x C x N] spike residuals (whitened)
% Optional parameters (key/value pairs) [default]:
%   lambda      Regularizer param in optimize()             [defer to reg_wt]
%   reg_wt      Relative weight to put on regularizer       [ 0.1 ]
%   K_add       Basis waveforms to add to each channel      [ 0 ]
%
% This calls self.optimize() to solve the following optimization problem:
%       minimize    ||Y - basis.reconst(X)||^2 + lambda*||basis-basis_old||^2
%     subject to    basis is channel-specific and channelwise orthonormal
% where the norms are defined in terms of the whitened inner product, and Y is
% the spike waveforms (Y = basis_old.reconst(spk_old)).
%
% By default, this sets lambda = reg_wt * ||spikes||^2/||basis||^2, but this can
% be overridden by explicitly specifying the <lambda> parameter.
%
% If K_add > 0, then ihis initializes the new basis waveforms using a singular
% value decomposition (SVD) of the given residual.

% Optional params
ip = inputParser();
ip.addParameter('lambda', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('reg_wt', 0.1, @isscalar);
ip.addParameter('K_add', 1, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Reconstruct the spike waveforms (also performs input checks)
spikes = self.reconstruct_spikes(basis, spk, resid);
basis_old = basis.basis;

% Define lambda if necessary
lambda = prm.lambda;
if isempty(lambda)
    % Use reg_wt to define lambda
    spk_norm = compute_spike_norm(self, spikes);
    basis_norm = compute_basis_norm(self, basis_old);
    lambda = prm.reg_wt * spk_norm^2/basis_norm^2;
end

% Expand the basis if desired
if prm.K_add > 0
    basis_svd = init_basis_svd(self, resid, spk.r, prm.K_add);
    basis_old = cat(2, basis_old, basis_svd); % [L x K+K_add x C]
end

% Call optimize()
[basis_new, spk_new] = self.optimize(spikes, ...
    'lambda',lambda, 'basis_prev',basis_old);

% Construct the SpikeBasis object
basis_obj = basis.copy_modify(basis_new);

% spk_new.t is currently spike offsets, so add these to the original spk.t
spk_new.shiftTimes(spk.t);

end


% --------------------------     Helper functions     --------------------------


function spk_norm = compute_spike_norm(self, spikes)
% Compute the norm of the given spike data (in the Q1 subspace)
%   spk_norm_sq = compute_spike_norm(self, spikes)
%
% Returns:
%   spk_norm    Norm of the given spikes (in the Q1 subspace)
% Required arguments:
%   spikes      [Lw x C x N] spike waveforms (whitened)
%
% This projects the given spikes onto the Q1 subspace before computing the norm,
% which ensures that we are only measuring the component of the residual that we
% can capture under our given constraint of finite support.
Q1 = self.whbasis.Q1;
[Lw,C,LC] = size(Q1); [~,~,N] = size(spikes);
spikes_Q1 = reshape(Q1,[Lw*C,LC])' * reshape(spikes, [Lw*C, N]);
spk_norm = norm(spikes_Q1,'fro');
spk_norm = gather(double(spk_norm));
end


function basis_norm = compute_basis_norm(self, basis)
% Compute the norm of the spike basis (in whitened space)
%   basis_norm = compute_basis_norm(self, basis)
%
% Returns:
%   basis_norm  Norm of the given spike basis (whitened)
% Required arguments
%   basis       [L x K x C] basis waveforms (raw)

% Since the basis is channel-specific (i.e. we can think of it as a [L*C x K*C]
% block diagonal matrix with C [L x K] blocks), we can measure its norm in
% either the Q1 or Q2 bases; they will produce equivalent results. So we'll use
% the Q2 basis; it's a little easier. wh_02 whitens the raw waveforms and
% expresses it in Q2 coordinates.
wh_02 = self.whbasis.wh_02;
basis_norm_sq = 0;
for c = 1:self.C
    basis_Q2_c = wh_02(:,:,c) * basis(:,:,c);
    basis_norm_sq = basis_norm_sq + norm(basis_Q2_c,'fro')^2;
end
basis_norm = sqrt(basis_norm_sq);
end


function basis_new = init_basis_svd(self, resid, spk_r, K_add)
% Initialize new basis components by performing a SVD of the residual
%   basis_new = init_basis_svd(self, resid, spk_r, K_add)
%
% Returns:
%   basis_new   [L x K_add x C] new basis waveforms (raw)
% Required arguments:
%   resid       [Lw x C x N] spike residuals (whitened waveforms)
%   spk_r       [N x 1] detected sub-sample shift (1..R) to remove
%   K_add       Number of new basis waveforms to initialize
whbasis = self.whbasis;
% Remove the sub-sample shift from the residuals (which requires unwhitening)
resid_unwh = whbasis.unwhiten(resid);           % [L x C x N] unwhitened
resid_unwh = whbasis.interp.shiftArr(resid_unwh, spk_r, true);
[L,C,N] = size(resid_unwh);
resid_unwh = reshape(resid_unwh, [L*C, N]);     % [L*C x N] unwh + unshifted
% Re-whiten the residuals and transform them into Q1 coordinates
resid_Q1 = whbasis.wh_01(:,:) * resid_unwh;     % [L*C x N] in Q1 basis
% Perform the SVD in Q2 coordinates (a channelwise-whitened space)
% We can reuse some optimize() helper functions
self.Y = resid_Q1;                              % [L*C x N] in Q1 basis
A = self.init_spkbasis(K_add);                  % [L x K_add x C] in Q2 basis
basis_new = self.convert_A_to_spkbasis(A);      % [L x K_add x C] non-whitened
self.Y = [];
end
