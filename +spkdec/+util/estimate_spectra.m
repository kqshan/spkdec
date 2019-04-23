function spect = estimate_spectra(data, varargin)
% Estimate the power spectra of the given signals
%   spect = estimate_spectra(data, ...)
%
% Returns:
%   spect       Struct with fields:
%     f           [F x 1] normalized frequency (cycles/sample)
%     mag         [F x C] signal magnitude (see below)
%     psd         [F x C] one-sided power spectral density (PSD)
% Required arguments:
%   data        DataSrc object. Assumed to be purely real (not complex)
% Optional parameters (key/value pairs) [default]:
%   N_fft       FFT size                                      [ 4096 ]
%   winfunc     Function handle to generate the FFT window    [ @hann ]
%   batch_size  Size of each data batch                       [ 8*N_fft ]
%   n_batch     Number of batches to average over             [ 32 ]
%
% The units of spect.psd are data_units^2/(cycles/sample), and is such that
% trapz(spect.f, spect.psd) == rms(data).^2, approximately. The signal magnitude
% (spect.mag) is the square root of the two-sided PSD (the "two-sided" PSD
% includes negative frequencies).

%% Input handling

errid_arg = 'spkdec:util:estimate_spectra:BadArg';

% Optional parameters
ip = inputParser();
ip.addParameter('N_fft', 4096, @isscalar);
ip.addParameter('winfunc', @hann, @(x) isa(x,'function_handle'));
ip.addParameter('batch_size', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('n_batch', 32, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Get the relevant dimensions
N = prm.N_fft;
assert(mod(N,2)==0, errid_arg, 'N_fft must be even');
B = prm.n_batch;
T_b = prm.batch_size;
if isempty(T_b), T_b = 8*N; end
assert(T_b >= N, errid_arg, 'batch_size must be >= N_fft');

% Construct the window
win = prm.winfunc(N);

% Not strictly necessary, just lazy
assert(mod(T_b,N)==0, errid_arg, 'batch_size must be a multiple of N_fft');
windows_per_batch = T_b / N;

%% Compute the PSD

% Read the data
overlap = N/2;
x = data.readRand('batch_size',T_b, 'n_batch',B, 'overlap',overlap);
assert(isreal(x), errid_arg, 'Complex data is not supported');
[T_b, C, B] = size(x);

% Compute the PSD using Welch's method
if B==1
    % There's only a single batch, so just use pwelch
    psd = pwelch(x, win, overlap, N, 'twosided');
    psd = psd * 2*pi; % pwelch() uses frequencies in rad/s
    psd = gather(double(psd)); % Force output to host double
else
    % Break each batch into windows of size N_fft
    assert(T_b == N * windows_per_batch);
    x = reshape(x, [N, windows_per_batch, C, B]);
    % Compute the PSD
    x = x .* win;      % Window
    x = fft(x);        % FFT
    x = abs(x).^2;     % Power
    x = double(x);     % Convert to double-precision
    x = mean(x,4);     % Average over batches
    x = mean(x,2);     % Average over windows within each batch
    x = x/sum(win.^2); % Normalize
    x = gather(x);     % Move to host (CPU)
    psd = reshape(x, [N C]);
end

%% Return in the desired format

% Only keep the positive frequencies
f = (0:N/2)' / N;

% One-sided PSD
psd_1 = psd(1:N/2+1,:);
psd_1(2:N/2,:) = psd_1(2:N/2,:) + flipud(psd(N/2+2:end,:));

% Magnitude (sqrt of psd)
mag = sqrt(psd(1:N/2+1,:));

% Output struct
spect = struct('f',f, 'mag',mag, 'psd',psd_1);

end
