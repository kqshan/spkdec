function basis = init_spkbasis(src, K, varargin)
% Initialize a spike basis by detecting threshold-crossings in whitened data
%   basis = init_spkbasis(src, K, ... )
%
% Returns:
%   basis       New SpikeBasis object
% Required arguments:
%   src         DataSrc object to read from
%   K           Number of spike basis waveforms per channel
% Parameters (key/value pairs) that you should probably set [default]:
%   whitener    Whitener object to use                      [ none ]
%   interp      Interpolator object to use                  [ none ]
%   L           Basis waveform length (#samples)            [defer to interp]
%   t0          Sample index (1..L) corresponding to t=0    [ 1 ]
% Other, more optional parameters:
%   det_quant   Spike detection quantile                    [ 0.995 ]
%   det_val     Spike detection threshold                   [use det_quant]
%   batch_size  Size (#samples) of each batch               [ 256k ]
%   n_batch     Number of randomly-selected batches         [ 32 ]
%   verbose     Print status updates to stdout              [ false ]
%   ...         Additional params forwarded to spkdec.SpikeBasis.make_basis()
%
% This detects spikes by looking for peaks in the norm (across channels) of the
% whitened data. If <det_quant> is used (as opposed to <det_val>), then the
% detection threshold will be set for each batch independently.
%
% Spikes are then extracted using <t0> to align the spike window to the detected
% peak, and the spike basis is initialized to minimize the reconstruction error
% on these detected spikes.


%% Deal with inputs

errid_dim = 'spkdec:util:init_spkbasis:DimMismatch';
errid_arg = 'spkdec:util:init_spkbasis:BadArg';

% Optional parameters
ip = inputParser();
ip.KeepUnmatched = true; ip.PartialMatching = false;
ip.addParameter('whitener', [], @(x) isempty(x) || isa(x,'spkdec.Whitener'));
ip.addParameter('interp', [], @(x) isempty(x) || isa(x,'spkdec.Interpolator'));
ip.addParameter('L', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('t0', 1, @isscalar);
ip.addParameter('det_quant', 0.995, @isscalar);
ip.addParameter('det_val', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('batch_size', 256*1024, @isscalar);
ip.addParameter('n_batch', 32, @isscalar);
ip.addParameter('verbose', false, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;
extra_args = ip.Unmatched;

% Default value for the whitener
C = src.C;
whitener = prm.whitener;
if isempty(whitener)
    whitener = spkdec.Whitener.no_whiten(C);
else
    assert(whitener.C==C, errid_dim, 'whitener.C must match given spikes');
end
W = whitener.W;
filt_delay = whitener.delay;

% Default value for the interpolator
interp = prm.interp;
if isempty(interp)
    L = prm.L;
    assert(~isempty(L), errid_arg, 'Either "interp" or "L" must be specified');
    interp = spkdec.Interpolator.no_interp(L);
else
    L = interp.L;
end
Lw = L + W - 1;


%% Detect spikes in batches

% Plan the batches
[batch_starts, batch_len] = src.planRand( ...
    'batch_size',prm.batch_size, 'n_batch',prm.n_batch);
assert(batch_len > 5*(W+L), errid_arg, ...
    'Batch size is too short for spike detection');
nBatch = length(batch_starts);

% Plan the spike extraction
t0 = prm.t0;
T = batch_len - (W-1);                      % Batch length after whitening
t_rel = (1-filt_delay:L+filt_delay)' - t0;  % [Lw x 1]
idx_rel = t_rel + T*(0:C-1);                % [Lw x C]

% Verbose output
verbose = prm.verbose;
if verbose, fprintf('Detecting spikes...'); end

% Detect spikes
spikes = cell(nBatch,1);
for ii = 1:nBatch
    % Read and whiten the data
    x = src.read(batch_starts(ii), batch_len);
    x = whitener.whiten(x);
    assert(size(x,1)==T);
    
    % Evaluate the detection metric
    det_metric = sqrt(sum(x.^2,2));
    
    % Determine the detection threshold (on a batch-by-batch basis)
    det_thresh = prm.det_val;
    if isempty(det_thresh)
        det_thresh = quantile(det_metric, prm.det_quant);
    end
    
    % Find spikes that satsify the following criteria:
    % 1. Exceeds the detection threshold
    is_spk = (det_metric > det_thresh);
    % 2. Is a local maximum
    delta = diff(det_metric);
    is_spk = is_spk & [true; delta > 0] & [delta < 0; true];
    % 3. Is a regional maximum in a radius of +/- L
    spk_t = find(is_spk);
    dt_refrac = L;
    is_spk = spkdec.Math.is_reg_max(spk_t, det_metric, dt_refrac);
    spk_t = spk_t(is_spk);
    % 4. Is not too close to the boundaries
    is_spk = (spk_t >= 1-t_rel(1)) & (spk_t <= T-t_rel(end));
    spk_t = spk_t(is_spk);
    
    % Extract spikes into a [Lw x C x nSpk] array
    N = length(spk_t);
    extract_idx = idx_rel(:) + spk_t';  % [Lw*C x N] indices
    spikes{ii} = reshape(x(extract_idx), [Lw C N]);
end

% Concatenate all the batches together
spikes = cat(3, spikes{:});
N = size(spikes, 3);
if verbose, fprintf('%d spikes detected\n',N); end


%% Initialize the basis waveforms

basis = spkdec.SpikeBasis.make_basis(spikes, K, 't0',t0, ...
    'whitener',whitener, 'interp',interp, 'verbose',verbose, extra_args);

end
