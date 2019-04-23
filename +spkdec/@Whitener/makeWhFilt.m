function [whfilt, wh_spec] = makeWhFilt(spect, varargin)
% Construct whitening filters for the given noise spectra
%   [whfilt, wh_spec] = Whitener.makeWhFilt(spect, varargin)
%
% Returns:
%   whfilt      [W x C] spike-band whitening filters
%   wh_spec     Struct of filter specifications
%     f           [F x 1] normalized frequency (cycles/sample)
%     wh_resp     [F x C] filter frequency response
%     wh_mag      [F x C] post-whitened noise magnitude
%     tgt_mag     [F x 1] desired post-whitened magnitude
% Required arguments:
%   spect       Noise spectrum
%     f           [F x 1] normalized frequency (cycles/sample)
%     mag         [F x C] noise magnitude (sqrt of two-sided PSD)
% Optional parameters (key/value pairs) [default]:
%   filt_len    Desired filter length (must be odd)         [ 201 ]
%   tgt_mag     Desired whitened magitude ([F x 1] or func) [ @(f) 1 ]
%   aa_freq     Anti-aliasing filter frequency              [ 0.47 ]
%   reg_lambda  Relative weight for regularizer term        [ 1 ]
%
% This designs real symmetric filters to approximate the desired response. These
% FIR filtres are designed using least-squares with a time-domain regularizer.
%
% Regarding aa_freq: The source data often have a high-frequency rolloff due to
% the acquisition system's anti-aliasing filter. This option prevents us from
% trying to whiten this rolloff by choosing to ignore frequencies above aa_freq.
%
% See also: spkdec.util.estimate_spectra

%% Deal with inputs 

errid_arg = spkdec.Whitener.errid_arg;

% Optional parameters
ip = inputParser();
ip.addParameter('filt_len', 201, @isscalar);
ip.addParameter('tgt_mag', @(f) ones(size(f)));
ip.addParameter('aa_freq', 0.47, @isscalar);
ip.addParameter('reg_lambda', 1, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Validate the inputs
% Input spectrum
freq = spect.f(:);
assert(freq(1)==0 && freq(end)==0.5 && all(diff(freq)>0), errid_arg, ...
    'spect.f must range from 0 to 0.5 and be monotonically increasing');
input_mag = spect.mag;
[F,C] = size(input_mag);
assert(all(input_mag > 0,'all'), errid_arg, ...
    'spect.mag must be strictly positive');
% Filter design
L = prm.filt_len;
assert(mod(L,2)==1, errid_arg, 'filter_len must be odd');
L_half = (L-1)/2;
lambda = prm.reg_lambda;
aa_mask = (freq <= prm.aa_freq);
% Target magnitude
tgt_mag = prm.tgt_mag;
if isa(tgt_mag,'function_handle')
    tgt_mag = tgt_mag(freq);
else
    assert(isnumeric(tgt_mag) && numel(tgt_mag)==F, errid_arg, ...
        'tgt_mag must be a [F x 1] vector or a function handle');
    tgt_mag = tgt_mag(:);
end
assert(all(tgt_mag >= 0), errid_arg, 'tgt_mag must be nonnegative');
% Make sure we have enough data
assert(sum(aa_mask) > L*3, errid_arg, ...
    ['length(spect.f) must be much greater than filt_len.\n' ...
     'As a rule of thumb, we are requiring F > 3*filt_len, where F ' ...
     'is the number of frequencies in spect.f that are < aa_freq']);

%% Design the filters

% Construct a matrix corresponding to the discrete cosine transform
dct_mat = cos(2*pi*freq * (0:L_half));
% Define the regularizer
reg = ((0:L_half)'./L_half).^2;
total_input_mag = sum(input_mag(aa_mask,:), 1); % [1 x C]
reg = reg/sum(reg) .* total_input_mag * lambda;
% Solve the least-squares problem for our decision variable x
x = zeros(L_half+1, C);
for c = 1:C
    A = [input_mag(aa_mask,c) .* dct_mat(aa_mask,:); diag(reg(:,c))];
    b = [tgt_mag(aa_mask); zeros(L_half+1,1)];
    x(:,c) = A \ b;
end
% Convert these into filter coefficients
whfilt = [flipud(x(2:end,:))/2; x(1,:); x(2:end,:)/2];

%% Evaluate the filter performance

if nargout >= 2
    % Evaluate the filter response
    filt_resp = dct_mat * x;
    wh_mag = filt_resp .* input_mag;
    % Create the struct
    wh_spec = struct('f',freq, 'wh_resp',filt_resp, ...
        'wh_mag',wh_mag, 'tgt_mag',tgt_mag);
end

end
