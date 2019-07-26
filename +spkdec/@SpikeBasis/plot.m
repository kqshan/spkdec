function plot(self, varargin)
% Plot these spike basis waveforms
%   plot(self, ...)
%
% Optional parameters (key/value pairs) [default]:
%   whiten    Whiten the basis waveforms                    [ false ]
%   interp    Apply sub-sample interpolation                [ true ]
%   tlim      Time axis display limits (sample indices)     [ 1, L ]
%   spacing   Y-axis spacing of waveforms                   [ auto ]
%   colors    [D x 3] line colors or colormap function      [@lines]

% Optional parameters
ip = inputParser();
ip.addParameter('whiten', false, @isscalar);
ip.addParameter('interp', true, @isscalar);
ip.addParameter('tlim', [1,self.L], @(x) numel(x)==2);
ip.addParameter('spacing', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('colors', @lines, ...
    @(x) isa(x,'function_handle') || isequal(size(x),[self.D,3]) );
ip.parse( varargin{:} );
prm = ip.Results;

% Dimensions
tlim = prm.tlim;
T = diff(tlim)+1;
L = self.L; C = self.C; D = self.D; R = self.R;
Lw = L + self.W-1;

% Obtain the [T x C x D] waveforms to plot (or [R*T x C x D] if interp=true)
ext_idx = (tlim(1):tlim(2))';
if prm.whiten
    ext_idx = ext_idx + self.whitener.delay;
    if prm.interp
        % whiten=true, interp=true
        wave = self.toKern();               % [Lw x C x D x R]
        % Need to flip the order of the sub-sample shifts
        wave = wave(ext_idx,:,:,end:-1:1);  % [T x C x D x R]
        wave = permute(wave,[4 1 2 3]);     % [R x T x C x D]
        wave = reshape(wave, [R*T,C,D]);    % [R*T x C x D]
    else
        % whiten=true, interp=false
        wave = self.whitener.toMat(L, 'flatten',true) ...
            * reshape(self.basis,[L*C,D]);  % [Lw*C x D]
        wave = reshape(wave, [Lw,C,D]);     % [Lw x C x D]
        wave = wave(ext_idx,:,:);           % [T x C x D]
    end
else
    if prm.interp
        % whiten=false, interp=true
        wave = reshape(self.basis,[L,C*D]); % [L x C*D]
        wave = self.interp.interp(wave);    % [R*L x C*D]
        wave = reshape(wave, [R,L,C*D]);    % [R x L x C*D]
        wave = wave(:,ext_idx,:);           % [R x T x C*D]
        wave = reshape(wave, [R*T,C,D]);    % [R*T x C x D]
    else
        % whiten=false, interp=false
        wave = self.basis;                  % [L x C x D]
        wave = wave(ext_idx,:,:);           % [T x C x D]
    end    
end
if prm.interp
    R_eff = R;
else
    R_eff = 1;
end
% Get the time coords too
wave_t = (tlim(1):tlim(2))' - self.t0;      % [T x 1]
wave_t = (wave_t + (-R_eff+1:0)/R_eff)';    % [R x T]
wave_t = wave_t(:);
RT = length(wave_t);                        % RT = prm.interp ? R*T : T

% Convert these into plot coordinates
% [(RT+1)*C x D] NaN-separated x-coordinates
x_centers = (1:D)' * T;
plot_x = repmat([wave_t; NaN], [1 C]);      % [RT+1 x C]
plot_x = plot_x(:) + x_centers';
xlims = [min(plot_x(:)), max(plot_x(:))] + [-1/R_eff, 0];
% Automatic y-spacing
spacing = prm.spacing;
if isempty(spacing)
    channel_gap = max(0,wave(:,2:end,:)) - min(0,wave(:,1:end-1,:));
    max_gap = max([0; channel_gap(:)]);
    spacing = 1.1 * max_gap;
end
y_centers = -(1:C)'*spacing;
% [(RT+1)*C x D] NaN-separated y-coordinates
plot_y = wave + y_centers';                 % [RT x C x D]
plot_y = [plot_y; NaN(1,C,D)];              % [RT+1 x C x D]
plot_y = reshape(plot_y, [(RT+1)*C,D]);
ylims = [min(plot_y(:)), max(plot_y(:))] + [-0.05,0.05]*spacing;

% Plot
figure();
% Draw the lines
lh = plot(plot_x, plot_y, '-');
% Color them
colors = prm.colors;
if isa(colors,'function_handle')
    colors = colors(D);
end
for d = 1:D
    lh(d).Color = colors(d,:);
end
% Label the axes
[yc_sorted,sortidx] = sort(y_centers);
set(gca, 'XLim',xlims, 'XTick',x_centers, 'XTickLabel',sprintfc('%d',1:D), ...
    'YLim',ylims, 'YTick',yc_sorted, 'YTickLabel',sprintfc('%d',sortidx) );
xlabel('Feature space dimensions'); ylabel('Data channels');

end