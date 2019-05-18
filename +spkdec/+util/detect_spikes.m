function detect_spikes(basis, source, outputs, varargin)
% Detect spikes using sparse deconvolution
%   detect_spikes(basis, source, outputs, ...)
%
% Required arguments:
%   basis       SpikeBasis object defining the basis waveforms and whitening
%   source      [T x C] raw data (DataSrc object)
%   outputs     Struct of DataSink objects for output (see below)
%
% Optional parameters (key/value pairs) [default]:
%   solver      Solver object to perform detection with     [spkdec.Solver()]
%   det_thresh  Spike detection threshold                   [defer to solver]
%   det_refrac  Spike detection refractory period           [defer to solver]
%   batch_size  Batch size for processing                   [ 1M ]
%   verbose     Print status updates to stdout              [ false ]
%   vb_period   Verbose update period (sec)                 [ 30 ]
%
% The <outputs> argument is a struct where each field is a DataSink object:
%   feature     [D x N] spike features (D = basis.K*basis.C)
%   index       [N] spike times (1..source.T) for spike center (basis.t0)
%   subidx      [N] sub-sample shift index (1..basis.R)
%   relnorm     [N] spike norm relative to the detection threshold
%   resid       [L+W-1 x C x N] whitened residual around each spike
%   data_resid  [T x C] data residual (raw data - detected spikes)
% Missing or empty output fields are ignored.
%
% Notes:
% * This will not detect spikes within the first 2*filt_delay + t0-1 samples or
%   the last 2*(W-filt_delay-1) + L-t0 samples of the data, where t0 = basis.t0
%   and filt_delay = basis.whitener.filt_delay
% * The spike times (outputs.index) correspond to the spike center. This differs
%   from the output of Solver.detect() or the input of SpikeBasis.conv_spk().
% * Normally, the sub-sample shifts specified by basis.interp will correspond to
%   backwards shifts by a fraction of a sample, and hence
%       spk_t = outputs.index + (outputs.subidx - 1)/basis.R
%   provides the spike time with temporal superresolution.

t_func_start = tic();
errid_pfx = 'spkdec:util:detect_spikes';

%% Deal with inputs

% Optional parameters
ip = inputParser();
ip.addParameter('solver', [], @(x) isempty(x) || isa(x,'spkdec.Solver'));
ip.addParameter('det_thresh', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('det_refrac', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('batch_size', pow2(20), @isscalar);
ip.addParameter('verbose', false, @isscalar);
ip.addParameter('vb_period', 30, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Get the solver object
solver = prm.solver;
if isempty(solver)
    solver = spkdec.Solver();
else
    % So that det_thresh and det_refrac don't produce unexpected changes in the
    % calling function. Also, copying a Solver object is very cheap.
    solver = copy(solver);
end
% Override the det_thresh and det_refrac
if ~isempty(prm.det_thresh)
    solver.det_thresh = prm.det_thresh;
end
if ~isempty(prm.det_refrac)
    solver.det_refrac = prm.det_refrac;
end

% Dimensions and other local variables
C = basis.C;    % Number of channels
D = basis.K*C;  % Number of feature space dimensions
L = basis.L;    % Waveform length (non-whitened)
W = basis.W;    % Whitening filter length
Lw = L + W-1;   % Waveform length (whitened)
whitener = basis.whitener;
filt_delay = whitener.delay;
% Spike offset = offset from the 1st sample of the spike to its center (t0)
spk_offset = basis.t0-1;
spk_offset_wh = spk_offset+filt_delay;

% Check the input
errid_dim = [errid_pfx ':DimMismatch'];
assert(source.hasShape([Inf C]), errid_dim, 'source.shape must be [Inf x C]');

% Check the output dimensions and make sure each has a field
output_check = {'feature',[D Inf]; 'index',[Inf]; 'subidx',[Inf]; ...
    'relnorm',[Inf]; 'resid',[Lw C Inf]; 'data_resid',[Inf C]}; %#ok<NBRAK>
for ii = 1:size(output_check,1)
    [fname, dims] = deal(output_check{ii,:});
    if ~isfield(outputs,fname)
        outputs.(fname) = [];
    end
    sink = outputs.(fname);
    assert(isempty(sink) || sink.hasShape(dims), errid_dim, ...
        'output.%s must have shape = %s', fname, mat2str(dims));
end
% Raise a warning if there are extra fields
extra_fields = setdiff(fieldnames(outputs), output_check(:,1));
if ~isempty(extra_fields)
    extra_fields = sprintf('%s ',extra_fields{:});
    warning([errid_pfx ':ExtraOutputs'], ...
        'Extra output field(s) will be ignored: %s', extra_fields);
end

% We need the non-whitened spike basis to compute the 'data_resid' output
if ~isempty(outputs.data_resid)
    basis_nonwh = basis.copy_nonWh();
end

%% Perform spike detection

% Verbose header
verbose = prm.verbose;
if verbose
    fprintf('detect_spikes() started at %s (C=%d, D=%d, R=%d, Lw=%d)\n', ...
        datestr(now(),31), C, D, basis.R, Lw);
    % An explanation of the columns:
    %   t_elapsed   Cumulative wall time since start of run
    %   batch#      Last batch # processed
    %   samp/ms     #data samples processed since last update, divided by
    %               runtime since last update (a measure of processing speed)
    %   nSpk        #spikes detected since last update
    %   /25ksamp    #spikes detected per 25,000 data samples (equivalent to
    %               spikes/second if the data were sampled at 25 kHz)
    vb_hdr = 't_elapsed  batch# | samp/ms |    nSpk  /25ksamp\n';
    fprintf(vb_hdr);
    nUpdate_since_hdr = 0;
    vb_fmt = '%9.1f  %6d | %7.1f | %7d  %8.2f\n';
    nSamp_since_update = 0;
    nSpk_since_update = 0;
    t_update = tic();
end

% Runtime and other stats
runtime = struct('load',0, 'detect',0, 'output',0);
t_start = tic();
nSamp_total = 0;
nSpk_total = 0;

% Perform the detection in batches
batch_no = 0;
is_last_batch = false;
while ~is_last_batch
    batch_no = batch_no + 1;
    is_first_batch = (batch_no == 1);
    
    % Get the whitened data for this batch ---------------------------------
    
    % Read the next bit of raw data
    [data_new, is_last_batch] = source.readNext(prm.batch_size);
    
    % Prepend the raw data carried over from the previous batch
    if is_first_batch
        data_offset = 0;
        data = data_new;
    else
        data_offset = data_offset + data_stride;
        data = [data_carryover; data_new];
    end
    % Save some raw data for the next batch (whitening filter overlap)
    data_stride = size(data,1) - (W-1);
    assert(data_stride > 0, errid_dim, 'Data is too short for detection');
    data_carryover = data(data_stride+1:end, :); % [W-1 x C]
    
    % Apply the whitening filter
    y = whitener.whiten(data);              % [data_stride x C]
    assert(size(y,1)==data_stride);
    y_offset = data_offset + filt_delay;    % Offset relative to <source>
    
    % Prepend the residual from the previous batch
    if ~is_first_batch
        y = [resid_carryover; y]; %#ok<AGROW>
        y_offset = y_offset - size(resid_carryover,1);
    end
    runtime.load = runtime.load + toc(t_start); t_start = tic();
    
    % Detect spikes in this batch ------------------------------------------
    
    % Perform the detection
    [spk, lims, resid] = solver.detect(basis, y, ...
        'trunc_1',false, 'trunc_2',~is_last_batch, ...
        'residuals',{'data','spk'});
    nSpk = spk.N;
    nSpk_total = nSpk_total + nSpk;
    runtime.detect = runtime.detect + toc(t_start); t_start = tic();
    
    % This only detected spikes in the range specified by <lims>
    % Save the leftover residual for the next batch to detect spikes in
    assert(lims(1)==1);
    nSamp = lims(2);
    nSamp_total = nSamp_total + nSamp;
    resid_carryover = resid.data(nSamp+1:end, :);
    
    % Generate output ------------------------------------------------------
    
    % Verbose output (runtime, spike count, etc)
    if verbose
        % Update counters
        nSamp_since_update = nSamp_since_update + nSamp;
        nSpk_since_update = nSpk_since_update + nSpk;
        t_since_update = toc(t_update);
        if t_since_update >= prm.vb_period
            % Replicate the header if it's been a while
            if nUpdate_since_hdr >= 40
                fprintf(vb_hdr);
                nUpdate_since_hdr = 0;
            end
            % Print update
            fprintf(vb_fmt, toc(t_func_start), batch_no, ...
                nSamp_since_update/(t_since_update*1e3), nSpk_since_update, ...
                nSpk_since_update/(nSamp_since_update/25e3));
            nUpdate_since_hdr = nUpdate_since_hdr + 1;
            % Reset counters
            nSamp_since_update = 0;
            nSpk_since_update = 0;
            t_update = tic();
        end
    end
    % These are taken (almost) directly from the detection output
    if ~isempty(outputs.feature)    % feature = spk.X
        outputs.feature.append(spk.X);
    end
    if ~isempty(outputs.index)      % index = spk.t + offsets
        outputs.index.append(spk.t + y_offset + spk_offset_wh);
    end
    if ~isempty(outputs.subidx)     % subidx = spk.r
        outputs.subidx.append(spk.r);
    end
    if ~isempty(outputs.resid)      % resid = resid.spk
        outputs.resid.append(resid.spk);
    end
    
    % Spike norm relative to the detection threshold
    if ~isempty(outputs.relnorm)
        spknorm = basis.spkNorms(spk);
        thresh = sqrt(solver.det_thresh * D); % See spkdec.Solver.det_thresh
        outputs.relnorm.append(spknorm / thresh);
    end
    
    % Non-whitened residual
    if ~isempty(outputs.data_resid)
        % Compute the non-whitened spike reconstuction -------------
        
        % Shift the spike times since these are non-whitened spikes
        spk_nonwh = copy(spk);
        spk_nonwh.shiftTimes(spk_offset_wh-spk_offset);
        % Perform the convolution (producing a [nSamp+Lw-1 x C] array)
        spike_rec = basis_nonwh.conv_spk(spk_nonwh, nSamp + W-1);
        % Add in the overlap from the previous batch
        if ~is_first_batch
            spike_rec(1:Lw-1,:) = spike_rec(1:Lw-1,:) + spike_rec_carryover;
        end
        % Defer the portion that overlaps with the next batch
        if ~is_last_batch
            spike_rec_carryover = spike_rec(nSamp+1:end,:);
            spike_rec = spike_rec(1:nSamp,:);
        end
        
        % Compute and write the residual ---------------------------
        
        % Align the raw data with the start of this reconstruction
        if is_first_batch
            % The first filt_delay samples are not used for spike detection
            outputs.data_resid.append(data_new(1:filt_delay, :));
            data_rec = data_new(filt_delay+1:end, :);
        else
            % Prepend the carryover
            data_rec = [data_rec_carryover; data_new];
        end
        % Truncate the end, saving it for the next batch
        rec_len = size(spike_rec,1);
        data_rec_carryover = data_rec(rec_len+1:end,:);
        data_rec = data_rec(1:rec_len,:);
        % Write the residual
        outputs.data_resid.append(data_rec - spike_rec);
        % Flush the carryover if this is the last batch
        if is_last_batch
            outputs.data_resid.append(data_rec_carryover);
        end
    end
    runtime.output = runtime.output + toc(t_start); t_start = tic();
end

% Verbose summary
if verbose
    fprintf('detect_spikes() completed at %s\n', datestr(now(),31));
    fprintf('  %d spikes detected over %.2fM samples\n', ...
        nSpk_total, nSamp_total/1e6);
    fprintf('  %.1f samples per spike or %.1f spikes per 25k samples\n', ...
        nSamp_total/nSpk_total, nSpk_total/(nSamp_total/25e3));
    for fn = fieldnames(runtime)'
        fprintf('%12s: %6.1f\n',fn{1},runtime.(fn{1}));
    end
    t_total = toc(t_func_start);
    fprintf('%12s: %6.1f sec\n', 'Total', t_total);
    fprintf('  Processed %.1fk samples/sec or %.1fx realtime if Fs=25kHz\n', ...
        nSamp_total/t_total/1e3, (nSamp_total/25e3)/t_total);
end

end
