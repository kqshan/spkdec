function [spk, lims, residuals] = detect_spikes_chunk(basis, data, varargin)
% Detect spikes in the given chunk of data
%   [spk, lims, residuals] = detect_spikes_chunk(basis, data, varargin)
%
% Returns:
%   spk         Spike times and features (Spikes object)
%   lims        [first,last] data index (1..T-V) where detection was attempted
%   residuals   Struct of residuals (fields determined by 'residual' parameter)
%     data        [T x C] residuals: data - basis.conv_spk(spikes.spk)
%     spk         [L+W-1 x C x N] spike-centered residuals
%     spk_unwh    [L x C x N] spike-centered residuals, unwhitened
%                 (see SpikeBasis.unwhiten)
% Required arguments:
%   basis       Spike waveform basis (SpikeBasis object)
%   data        [T x C] whitened data to detect spikes in
%
% Optional parameters (key/value pairs) [default]:
%   det_thresh  Spike detection threshold (rel to K*C)          [ 10 ]
%   det_refrac  Spike detection refractory period (#samples)    [ 0 ]
%   trunc_1     Truncate detection at the left boundary         [ false ]
%   trunc_2     Truncate detection at the right boundary        [ false ]
%   residuals   Names of residuals to compute and return        [{'data'}]
%   solver_args Arguments to forward to the Solver constructor  [ {} ]
%
% Some things to be aware of:
% * Spike times
%   To keep things simple, this uses purely causal convolutions. This means that
%   the spike times (spk.t) correspond to the start of the whitened waveform.
%   If you want spike times for the center of the spike instead, then you should
%   use spk_times = spk.t + basis.whitener.delay + (basis.t0-1).
% * det_thresh
%   Spike detection is ultimately performed by spkdec.Solver, which seeks to
%   minimize ||data-basis.conv(spk)||^2 + beta*spk.N, and this beta is related
%   to det_thresh by beta = det_thresh * K * C
% * trunc_1, trunc_2
%   Data boundaries are problematic because they may contain partial spikes.
%   These truncation options remove spikes that were detected within V samples
%   of the boundary (and any spikes within V samples of those, and so on), with
%   a maximum extent of 4*V.

% Get some dimensions
[T,C] = size(data);
V = basis.V;

% Optional parameters
ip = inputParser();
ip.addParameter('det_thresh', 10, @isscalar);
ip.addParameter('det_refrac', 0, @isscalar);
ip.addParameter('trunc_1', false, @isscalar);
ip.addParameter('trunc_2', false, @isscalar);
ip.addParameter('residuals', {'data'}, @iscellstr);
ip.addParameter('solver_args', {}, @iscell);
ip.parse( varargin{:} );
prm = ip.Results;

% Detect spikes ----------------------------------------------------------------

% Perform the deconvolution
solver = spkdec.Solver('det_refrac',prm.det_refrac, prm.solver_args{:});
beta = prm.det_thresh * basis.K * basis.C;
[spk, data_resid] = solver.solve(basis, data, beta);

% Truncate results
tau = V;
% Find the start of the chunk
if prm.trunc_1
    spk_t_aug = [0; spk.t; Inf];
    first_ok = find(diff(spk_t_aug) >= tau, 1, 'first');
    t1 = spk_t_aug(first_ok) + tau + 1;
    t1 = min(t1, 1 + 4*tau);
else
    t1 = 1;
end
% Find the end of the chunk
Tx = T - V;
if prm.trunc_2
    spk_t_aug = [-Inf; spk.t; Tx+1];
    last_ok = find(diff(spk_t_aug) >= tau, 1, 'last') + 1;
    t2 = spk_t_aug(last_ok) - tau - 1;
    t2 = max(t2, Tx - 4*tau);
else
    t2 = Tx;
end
% Perform the truncation
lims = [t1 t2];
mask = (spk.t >= t1) & (spk.t <= t2);
if ~all(mask)
    spk = spk.subset(mask);
    % Update the residuals
    data_resid = data - basis.conv_spk(spk, Tx);
end

% Compute the residuals --------------------------------------------------------

% See what we need
if nargout < 3, return; end
resid_names = prm.residuals;
residuals = struct();
% Data residual
if ismember('data',resid_names)
    residuals.data = data_resid;
end
% Spike-centered whitened residuals
if any(startsWith(resid_names,'spk'))
    N = spk.N;
    extract_idx = (0:V)' + spk.t';              % [L+W-1 x N]
    spk_resid = data_resid(extract_idx(:), :);  % [(L+W-1)*N x C]
    spk_resid = reshape(spk_resid,[V+1, N, C]); % [L+W-1 x N x C]
    spk_resid = permute(spk_resid,[1 3 2]);     % [L+W-1 x C x N]
end
if ismember('spk',resid_names)
    residuals.spk = spk_resid;
end
% Unwhitened
if ismember('spk_unwh',resid_names)
    residuals.spk_unwh = basis.unwhiten(spk_resid);
end

end
