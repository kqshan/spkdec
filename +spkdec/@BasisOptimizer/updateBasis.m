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
%   D_add       Number of basis waveforms to add            [ 0 ]
%
% This calls self.optimize() to solve the following optimization problem:
%       minimize    ||Y - basis.reconst(X)||^2 + lambda*||basis-basis_old||^2
%     subject to    basis is orthonormal
% where the norms are defined in terms of the whitened inner product, and Y is
% the spike waveforms (Y = basis_old.reconst(spk_old) + resid).
%
% By default, this sets lambda = reg_wt * ||spikes||^2/||basis||^2, but this can
% be overridden by explicitly specifying the <lambda> parameter.
%
% If D_add > 0, then ihis initializes the new basis waveforms using a singular
% value decomposition (SVD) of the given residual.

% Optional params
ip = inputParser();
ip.addParameter('lambda', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('reg_wt', 0.1, @isscalar);
ip.addParameter('D_add', 0, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Reconstruct the spike waveforms (also performs input checks)
spikes = self.reconstruct_spikes(basis, spk, resid);

% Define lambda if necessary
lambda = prm.lambda;
if isempty(lambda)
    % Use reg_wt to define lambda
    spk_norm = compute_spike_norm(self, spikes);
    basis_norm = compute_basis_norm(self, basis.basis);
    lambda = prm.reg_wt * spk_norm^2/basis_norm^2;
end
basis_prev = basis.basis;

% Expand the basis
D_add = prm.D_add;
if (D_add > 0)
    % Remove the detected sub-sample shift from the residual and represent the
    % residual in Q1 coordinates
    resid_Q1 = transform_residual(self, resid, spk.r);
    % Initialize the new axes
    self.Y = resid_Q1;
    A_new = self.init_spkbasis(D_add);
    self.Y = [];
    % Append this to the old basis and re-orthonormalize
    A_old = self.convert_spkbasis_to_A(basis_prev);
    A = self.append_bases(A_old, A_new);
    % Convert back to the non-whitened space
    basis_prev = self.convert_A_to_spkbasis(A);
end

% Optimize the basis
[basis_new, spk_new] = self.optimize(spikes, ...
    'lambda',lambda, 'basis_prev',basis_prev);

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
%   basis       [L x C x D] spike basis waveforms (non-whitened)
[L,C,D] = size(basis);
basis_Q1 = self.whbasis.wh_01(:,:) * reshape(basis,[L*C,D]);
basis_norm = norm(basis_Q1(:));
end


function resid_Q1 = transform_residual(self, resid, spk_r)
% Transform the given residuals into Q1 coordinates and remove sub-sample shift
%   resid_Q1 = transform_residual(self, resid, spk_r)
%
% Returns:
%   resid_Q1    [L x C x N] unshifted spike residuals in Q1 coordinates
% Required arguments:
%   resid       [Lw x C x N] whitened spike residuals
%   spk_r       [N x 1] detected sub-sample shift (1..R) to remove
whbasis = self.whbasis;
% Remove the sub-sample shift from the residuals (which requires unwhitening)
resid_unwh = whbasis.unwhiten(resid);           % [L x C x N] unwhitened
resid_unwh = whbasis.interp.shiftArr(resid_unwh, spk_r, true);
[L,C,N] = size(resid_unwh);
resid_unwh = reshape(resid_unwh, [L*C, N]);     % [L*C x N] unwh + unshifted
% Re-whiten the residuals and transform them into Q1 coordinates
resid_Q1 = whbasis.wh_01(:,:) * resid_unwh;     % [L*C x N] in Q1 basis
end
