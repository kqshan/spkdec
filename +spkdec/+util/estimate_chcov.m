function ch_cov = estimate_chcov(data, varargin)
% Estimate the cross-channel covariance of the given multi-channel signal
%   ch_cov = estimate_chcov(data, ...)
%
% Returns:
%   ch_cov      [C x C] estimated cross-channel noise covariance
% Required arguments:
%   data        DataSrc object
% Optional parameters (key/value pairs) [default]:
%   filter      [W x 1] or [W x C] data filter to apply     [ none ]
%   cap_val     Data value to truncate the source data to   [use cap_quant]
%   cap_quant   Data quantile to truncate the data to       [ 0.99 ]
%   batch_size  Size (#samples) of each batch               [ 32k ]
%   n_batch     Number of randomly-selected batches         [ 32 ]
%
% Some notes on how this estimates the covariance:
% * It actually computes the 2nd moment (i.e. src'*src/T), not the covariance
%   (which would imply subtracting the mean first).
% * In a crude attempt to make this more robust to high-amplitude outliers (such
%   as spikes), we truncate the data: data = max(-cap_val,min(cap_val,src)).
%   This causes us to underestimate the true covariance, but works well enough.
%   Note that cap_val may be a scalar or a [C x 1] vector, and must be > 0

errid_dim = 'spkdec:util:estimate_chcov:DimMismatch';
errid_arg = 'spkdec:util:estimate_chcov:BadArg';

% Optional parameters
ip = inputParser();
ip.addParameter('filter', []);
ip.addParameter('cap_val', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('cap_quant', 0.99, @isscalar);
ip.addParameter('batch_size', 32*1024, @isscalar);
ip.addParameter('n_batch', 32, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Read the data
x = data.readRand('batch_size',prm.batch_size, 'n_batch',prm.n_batch);
[T_b, C, B] = size(x);

% Apply the whitening filter
% Check its dimensions
whfilt = prm.filter;
if isempty(whfilt)
    pad_wh = 0;
else
    if size(whfilt,2)==1, whfilt = repmat(whfilt,[1 C]); end
    assert(size(whfilt,2)==C, errid_dim, 'filter must be [W x 1] or [W x C]');
    pad_wh = size(whfilt,1) - 1;
end
% Perform the filtering
for b = 1:B
    if isempty(whfilt), break; end
    x(:,:,b) = fftfilt(whfilt, double(x(:,:,b)));
end
% Remove any data that overlaps the boundaries
x = x(pad_wh+1:end,:,:);
T_b = T_b - pad_wh; assert(T_b > 0);

% Truncate the high-amplitude "outliers"
% Default value for cap_val (based on cap_quant)
cap_val = prm.cap_val;
if isempty(cap_val)
    abs_x = reshape(permute(abs(x),[1 3 2]), [T_b*B, C]);
    cap_val = quantile(abs_x, prm.cap_quant);
else
    assert(isscalar(cap_val) || numel(cap_val) == C, errid_dim, ...
        'If provided, cap_val must be a scalar or a [C x 1] vector');
    cap_val = cap_val(:)';
end
assert(all(cap_val > 0), errid_arg, 'cap_val must be strictly positive');
% Perform the truncation
x = max(-cap_val,min(cap_val, x));

% Compute the 2nd moment
ch_cov = zeros(C,C);
for b = 1:B
    ch_cov = ch_cov + gather(double(x(:,:,b)' * x(:,:,b)));
end
ch_cov = ch_cov / (T_b * B);

end
