function [wh, specs] = make_spk_whitener(data, varargin)
% Create a Whitener object for spike detection in the given data source
%   [wh, specs] = make_spk_whitener(data, ...)
%
% Returns:
%   wh          Whitener object representing the spike filter and whitener
%   specs       Struct of whitener specifications:
%     f           [F x 1] normalized frequency (cycles/sample)
%     raw_mag     [F x C] estimated magnitude spectra (sqrt of PSD) of raw data
%     wh_resp     [F x C] whitening filter frequency response
%     wh_mag      [F x C] magnitude spectra after whitening (= resp .* mag_raw)
%     tgt_mag     [F x 1] desired magnitude spectra (used in filter design)
%     ch_cov      [C x C] estimated cross-channel covariance of filtered data
%
% Required arguments:
%   data        [Inf x C] DataSrc object to read raw data from
%
% Optional parameters (key/value pairs) [default]:
%   N_fft       FFT size used to design whitening filter        [ 4096 ]
%   n_batch     Number of data batches used in estimation       [ 32 ]
%   filt_len    Desired filter length (must be odd)             [ 201 ]
%   bp_freq     Spike band cutoff freqs (cycles/sample)         [ 0.02, 0.4 ]
%   bp_order    Spike bandpass rolloff order (1 = 20dB/dec)     [ 2, 4 ]
%   min_resp    Minimum response of the designed bandpass       [ 0 ]
%   aa_freq     Anti-aliasing filter frequency                  [ 0.47 ]
%   reg_lambda  Relative weight for regularizer term            [ 1 ]
%
% What I'm calling the "magnitude spectra" is the square root of the two-sided
% power spectral density (PSD). See spkdec.util.estimate_spectra for more info.
%
% The bandpass filter specifications (bp_freq, bp_order) define the desired
% magnitude spectra for the filtered data. For a data sampling rate of 25 kHz,
% the default cutoffs correspond to 500 Hz and 10 kHz. You may need to adjust
% this (and/or the rolloff order) to suit your data.
%
% <min_resp> puts a floor on the desired magnitude spectra for the filtered
% data. If the specified bandpass filter has wide stopbands, then this may need
% to be set to a small value, such as 0.01, to prevent the whitening operation
% from becoming too poorly-conditioned.
%
% <aa_freq> and <reg_lambda> control the least-squares design of the symmetric
% FIR filters that will be whitening filters; see spkdec.Whitener.makeWhFilt()
% for more detail.
%
% See also: spkdec.util.estimate_spectra, spkdec.util.estimate_chcov,
% spkdec.Whitener.makeWhFilt

errid_arg = 'spkdec:util:make_spk_whitener:BadArg';

%% Deal with inputs

% Optional parameters
ip = inputParser();
ip.addParameter('N_fft', 4096, @isscalar);
ip.addParameter('n_batch', 32, @isscalar);
ip.addParameter('filt_len', 201, @isscalar);
ip.addParameter('bp_freq', [0.02, 0.4], @(x) numel(x)==2);
ip.addParameter('bp_order', [2, 4], @(x) numel(x)==2);
ip.addParameter('min_resp', 0, @isscalar);
ip.addParameter('aa_freq', 0.47, @isscalar);
ip.addParameter('reg_lambda', 1, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Validate the filter specs
L = prm.filt_len;
f1 = prm.bp_freq(1);
f2 = prm.bp_freq(2);
assert(f1 <= 0.5 && f2 <= 0.5, errid_arg, ['bp_freq must be <= 0.5 ' ...
    '(These should be given in units of cycles/sample)']);
assert(f2 > f1, errid_arg, 'bp_freq must be increasing');
n1 = prm.bp_order(1);
n2 = prm.bp_order(2);
assert(n1 >= 0 && n2 >= 0, errid_arg, 'bp_order must be >= 0');
% Make sure we have enough data
N = prm.N_fft;
assert(L < N/3, errid_arg, ['filt_len must be much less than N_fft.' ...
    '\nAs a rule of thumb, we are requiring filt_len < N_fft/3']);

% Make sure we have a [T x C] data source (and not inadvertently transposed)
src_shape = data.shape;
assert(length(src_shape)==2 && isinf(src_shape(1)) && ~isinf(src_shape(2)), ...
    errid_arg, 'Given data.shape must be [Inf x C]');

%% Do the thing

% Estimate the signal spectra (for each channel independently)
spectra = spkdec.util.estimate_spectra(data, 'N_fft',N, 'n_batch',prm.n_batch);

% Parse the filter specs into a desired frequency response
highpass = @(f) 1 - 1./(1 + (f/f1).^n1);
lowpass  = @(f)     1./(1 + (f/f2).^n2);
a = prm.min_resp;
tgt_mag  = @(f) a + (1-a)*highpass(f) .* lowpass(f);

% Design the filter
[wh_filt, specs] = spkdec.Whitener.makeWhFilt(spectra, ...
    'filt_len',L, 'tgt_mag',tgt_mag, ...
    'aa_freq',prm.aa_freq, 'reg_lambda',prm.reg_lambda);

% Estimate the cross-channel covariance of the filtered data
ch_cov = spkdec.util.estimate_chcov(data, ...
    'filter',wh_filt, 'n_batch',prm.n_batch);

% Design the cross-channel whitener
% I like the "sqrtm" method because it keeps the wh_ch matrix dominated by its
% diagonal, which means that each channel more or less stays put. "eig" for sure
% does not achieve this, and "chol" can also start to look a little weird (and
% there's not much advantage to having a triangular matrix in this application)
wh_ch = spkdec.Whitener.makeWhCh(ch_cov, 'method','sqrtm');


%% Assemble the output

% Construct the whitener
wh = spkdec.Whitener('wh_filt',wh_filt, 'delay',(L-1)/2, 'wh_ch',wh_ch);

% Add the additional fields to the specs struct
assert(isequal(specs.f, spectra.f));
specs.raw_mag = spectra.mag;
specs.ch_cov = ch_cov;

end
