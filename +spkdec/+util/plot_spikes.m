function handles = plot_spikes(spk, basis, data, varargin)
% Plot the selected spikes for visualization
%   handles = plot_spikes(spk, basis, data, varargin)
%
% Returns:
%   handles     Struct of handles to graphics objects (see below)
%
% Required arguments:
%   spk         Spike times and features to plot (Spikes object)
%   basis       Spike basis waveforms (SpikeBasis object)
%   data        [Inf x C] raw data (DataSrc object)
%
% Optional parameters (key/value pairs) [default]:
%   ah          Axes handle to plot in                      [ new ]
%   tlim        [first,last] sample index (1..T) to plot    [-Inf,Inf]
%   whiten      Show the data in whitened space             [ false ]
%   is_resid    <data> is residual, not original raw data   [ false ]
%   zoom_t      Time axis zoom factor on individual spikes  [ 4 ]
%   zoom_y      Y axis zoom factor on individual spikes     [ 1.5 ]
%   spacing     Plot spacing (data units)                   [ auto ]
%   Fs          Sample frequency (Hz) for time axis in ms   [ none ]

%% Deal with inputs

errid_arg = 'spkdec:util:plot_spikes:BadArg';
errid_dim = 'spkdec:util:plot_spikes:DimMismatch';

% Parse inputs
ip = inputParser();
ip.addParameter('ah', [], @(x) isempty(x) || ishandle(x));
ip.addParameter('tlim', [-Inf, Inf], @(x) numel(x)==2);
ip.addParameter('whiten', false, @isscalar);
ip.addParameter('is_resid', false, @isscalar);
ip.addParameter('zoom_t', 4, @isscalar);
ip.addParameter('zoom_y', 1.5, @isscalar);
ip.addParameter('spacing', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('Fs', [], @(x) isempty(x) || isscalar(x));
ip.parse( varargin{:} );
prm = ip.Results;

% Check the data
assert(isa(data,'spkdec.DataSrc'), errid_arg,'<data> must be a DataSrc object');
C = basis.C;
assert(data.hasShape([Inf C]), errid_dim, ...
    'data.shape must be [Inf x C] where C = basis.C');
data_T = data.len;

%% Truncate the spikes and data to the specified time limits

% Identify the data range to plot
plot_t1 = max(prm.tlim(1), 1);
plot_t2 = min(prm.tlim(2), data_T);
plot_offset = plot_t1-1;
plot_T = plot_t2 - plot_offset;

% Truncate the data (and apply whitening if desired)
if prm.whiten
    % Read an extra W-1 samples that will discarded during whitening
    whitener = basis.whitener;
    data = data.read(plot_t1-whitener.delay, plot_T+whitener.W-1);
    % Whiten it
    data = whitener.whiten(data);
else
    % Read exactly what we need
    data = data.read(plot_t1, plot_T);
    % Switch the spike basis to a non-whitened version
    basis = basis.copy_nonWh();
end
data = gather(data);

% Get some spike waveform dimensions
filt_delay = basis.whitener.delay;
L = basis.L;                            % Non-whitened spike waveform length
Lw = L + basis.W-1;                     % Whitened spike waveform length
spk_ctr_idx = basis.t0 + filt_delay;    % Index (1..Lw) of spike center index
spk_t1_offset = 1 - spk_ctr_idx;        % Spike first sample rel. to center
spk_t2_offset = Lw - spk_ctr_idx;       % Spike last sample rel. to center

% Truncate the spikes to those that overlap our plot range
spk_mask = (spk.t + spk_t2_offset >= plot_t1) ...
    & (spk.t + spk_t1_offset <= plot_t2);
spk = spk.subset(spk_mask);
spk.shiftTimes(-plot_offset);

%% Reconstruct the spikes

% Perform the spike convolution
spk.shiftTimes(spk_t2_offset);                  % Need to ensure spk.t >= 1
spk_conv = basis.conv_spk(spk, plot_T+Lw-1);    % [plot_T+2*(Lw-1) x C]
spk_conv = spk_conv(Lw:plot_T+Lw-1, :);         % [plot_T x C]
spk.shiftTimes(-spk_t2_offset);                 % Undo the earlier shift

% Further truncate the spikes to those within the plot range
spk_mask = (spk.t >= 1) & (spk.t <= plot_T);
spk = spk.subset(spk_mask);

% [plot_T x C] time series of the given data and the residual
if prm.is_resid
    resid = data;
    data = data + spk_conv;
else
    resid = data - spk_conv;
end

% Reconstruct individual spikes
spike_y = basis.reconst(spk);       % [Lw x C x N] spike reconstructions
rel_t = (1:Lw)' - spk_ctr_idx;

% The whitened spikes can be kinda long and uninteresting out in the tails, so
% truncate this to a length of about L instead
pad = 2;
mask = (rel_t >= (1-basis.t0)-pad) & (rel_t <= (L-basis.t0)+pad);
spike_y = spike_y(mask,:,:);
rel_t = rel_t(mask);

%% Lay out the plotting coordinates

% Lay out the individual spikes, appling the time-axis zoom
spk_t = double(spk.t);
[spike_t, spk_dt, zoom_t] = layout_spikes(spk_t, rel_t, prm.zoom_t, plot_T);
% Apply the y-axis zoom
zoom_y = prm.zoom_y;
spike_y = spike_y * zoom_y;

% Convert the spikes into plottable [(Lw+1)*N x C] with NaN separators
[Lw_disp, ~, N] = size(spike_y);
spike_y = permute(spike_y, [1 3 2]);            % [Lw x N x C]
spike_y = [spike_y; NaN(1,N,C)];                % [Lw+1 x N x C]
spike_y = reshape(spike_y,[(Lw_disp+1)*N, C]);  % [(Lw+1)*N x C]
% Same with the corresponding spike times
spike_t = [spike_t; NaN(1,N)];                  % [Lw+1 x N]
spike_t = spike_t(:);                           % [(Lw+1)*N x 1]

% Vertical layout
% Data and spike limits for each channel
data_min = min(data,[],1);
data_max = max(data,[],1);
spike_min = min([spike_y; zeros(1,C)], [], 1);
spike_max = max([spike_y; zeros(1,C)], [], 1);
% Default value for inter-channel spacing
spacing = prm.spacing;
if isempty(spacing)
    ch_range = data_max(C) - data_min(1);
    if (C > 1)
        ch_gap = max(diff(data,1,2), [],'all');
    else
        ch_gap = 0;
    end
    spacing = (ch_range + (C-1)*ch_gap) / C;
    spacing = 1.2 * spacing;
end
% Get the offsets between channels and between data and spikes
data_ch_offsets = (0:C-1) * spacing;
data_spike_offset = data_max(C) + data_ch_offsets(C) + spacing - spike_min(1);
spike_ch_offsets = zoom_y*data_ch_offsets + data_spike_offset;

% Construct the connector lines with temporal superresolution
spk_dt_r = double(spk.r-1) / basis.R;
conline_t1 = spk_t + spk_dt_r;
conline_t2 = spk_t + spk_dt + zoom_t*spk_dt_r;
conline_t = [conline_t1, conline_t1, conline_t2, conline_t2]';  % [4 x N]
conline_t = [conline_t; NaN(1,N)];                              % [5 x N]
conline_t = conline_t(:);                                       % [5*N x 1]
% And the y-coordinates
conline_padding = 0.1 * spacing;
conline_y0 = [
    data_min(1)  + data_ch_offsets(1)  - conline_padding
    data_max(C)  + data_ch_offsets(C)  + conline_padding
    spike_min(1) + spike_ch_offsets(1) - conline_padding
    spike_max(C) + spike_ch_offsets(C) + conline_padding
    ];
conline_y = repmat([conline_y0; NaN], [1 N]);
conline_y = conline_y(:);                                       % [5*N x 1]

% Convert time coordinates to seconds if desired
data_t = (1:plot_T)';
Fs = prm.Fs / 1e3;
if ~isempty(Fs)
    data_t = data_t/Fs;
    spike_t = spike_t/Fs;
    conline_t = conline_t/Fs;
end 

%% Plot

handles = struct();

% Axes to plot in
ah = prm.ah;
if isempty(ah)
    figure();
    ah = axes();
end
set(ah, 'Box','on', 'Layer','top', 'NextPlot','add', ...
    'XLim',[0, data_t(end)], 'YLim',conline_y0([1 end]));
handles.ah = ah;

% Connector lines
handles.connector = plot(ah, conline_t, conline_y, '-', ...
    'LineWidth',0.25, 'Color',[0.8 0.8 0.8]);
% Raw data and residuals
handles.data  = plot(ah, data_t,  data + data_ch_offsets, '-');
handles.resid = plot(ah, data_t, resid + data_ch_offsets, 'k-');
% Individual spikes
handles.spikes = plot(ah, spike_t, spike_y + spike_ch_offsets, '-');
if (N > 0)
    for c = 1:C
        handles.spikes(c).Color = handles.data(c).Color;
    end
end

% Make it pretty
if isempty(Fs)
    xlabel('Time (sample #)');
else
    xlabel('Time (ms)');
end
set(ah, 'YTick',[data_ch_offsets, spike_ch_offsets], ...
    'YTickLabel',sprintfc('ch %d',[1:C, 1:C]));


end



% -----------------------     Helper functions     -----------------------------


function [spike_t, spk_dt, zoom] = layout_spikes(spk_t, rel_t, zoom, T)
% Layout the spike time coordinates, zooming in on the x-axis
%   [spike_t, spk_dt, zoom] = layout_spikes(spk_t, rel_t, zoom, T)
%
% Returns:
%   spike_t     [L x N] spike waveform sample times (1..T) with zoom
%   spk_dt      [N x 1] offset from spk_t to spike_t at the spike center
% Required arguments:
%   spk_t       [N x 1] spike center times (1..T)
%   rel_t       [L x 1] sample times for the spike waveform
%   zoom        Desired zoom level (actual zoom may be lower)
%   T           Data range (will ensure that spike_t stays in the range [1 T])
%
% This preserves the relative timing of overlapping spikes and implements the
% zoom by shrinking the gaps between sequences of overlapping spikes.

% Get dimensions and validate some assumptions that we make
L = length(rel_t); assert(all(diff(rel_t)==1));
t0 = find(rel_t==0); assert(isscalar(t0));
N = length(spk_t);
if (N==0), spike_t = zeros(L,0); spk_dt = zeros(0,1); return; end

% Identify sequences of overlapping spikes
is_1st_in_seq = [true; diff(spk_t) >= L];
seq_first = find(is_1st_in_seq);            % First spike (1..N) in sequence
S = length(seq_first); 
seq_last = [seq_first(2:S)-1; N];           % Last spike (1..N) in sequence
% Sequence length (#samples) and gap (#samples) between sequences
seq_len = spk_t(seq_last) - spk_t(seq_first) + L;           % [S x 1]
seq_gap = spk_t(seq_first(2:S)) - spk_t(seq_last(1:S-1));   % [S-1 x 1]
% Match these up to the spikes
spk_seq = cumsum(is_1st_in_seq);            % Sequence # for each spike

% Find the maximum zoom level that would work
total_seq_len = sum(seq_len);
max_zoom = T / total_seq_len;
zoom = min(zoom, 0.9 * max_zoom);           % Leave us 10% leeway

% We wish to solve the following problem:
%     minimize  Amount that each spike has been moved
%   subject to  Relative timing is preserved within each sequence
%               Disjoint sequences do not overlap
%               Spikes lie fully within data range
%
% We can express this as the following LP in (spk_dt, seq_dt)
%     minimize  sum(abs(spk_dt))
%   subject to  spk_t + spk_dt == zoom*spk_t + seq_dt(spk_seq)
%               spk_dt(seq_last(k))-spk_dt(seq_first(k+1)) <= seq_gap(k)-zoom*L
%               spk_t(1) + spk_dt(1) + zoom*(1-t0) >= 1
%               spk_t(end)+spk_dt(end)+zoom*(L-t0) <= T

% In order to express this in standard form, however, we'll need to introduce
% slack variables such that spk_dt = spk_dt_pos - spk_dt_neg.
% x = [spk_dt_pos; spk_dt_neg; seq_dt]

% f'*x = sum(abs(spk_dt))
f = [ones(2*N,1); zeros(S,1)];
% A_eq*x = spk_dt - seq_dt(spk_seq) == (zoom-1)*spk_t = b_eq
A_eq = [eye(N), -eye(N), -full(sparse(1:N, spk_seq, 1, N, S))];
b_eq = (zoom-1) * spk_t;
% A1*x = spk_dt(seq_last(1:S-1)) - spk_dt(seq_first(2:S)) <= seq_gap-zoom*L = b1
M_s1 = full(sparse(1:S-1, seq_last(1:S-1), 1, S-1, N));
M_s2 = full(sparse(1:S-1, seq_first(2:S),  1, S-1, N));
A1 = [M_s1-M_s2, M_s2-M_s1, zeros(S-1,S)];
b1 = seq_gap - zoom*L;
% A2*x = -spk_dt(1) <= spk_t(1) + zoom*(1-t0) - 1 = b2
A2 = [-1, zeros(1,N-1), 1, zeros(1,N-1), zeros(1,S)];
b2 = spk_t(1) + zoom*(1-t0) - 1;
% A3*x = spk_dt(end) <= -spk_t(end) - zoom*(L-t0) + T = b3
A3 = [zeros(1,N-1), 1, zeros(1,N-1), -1, zeros(1,S)];
b3 = -spk_t(end) - zoom*(L-t0) + T;
% Concatenate the A,b
A = [A1; A2; A3];
b = [b1; b2; b3];
% lb <= x <= ub
lb = [zeros(2*N,1); -Inf(S,1)];
ub = Inf(2*N+S,1);

% Solve for x = [spk_dt_pos; spk_dt_neg; seq_dt]
opts = optimoptions('linprog', 'Display','none');
x = linprog(f, A, b, A_eq, b_eq, lb, ub, opts);

% Extract spk_dt and construct the full [L x N] spike time coordinates
spk_dt = x(1:N) - x(N+1:2*N);
spike_t = zoom*rel_t + (spk_t + spk_dt)';

end
