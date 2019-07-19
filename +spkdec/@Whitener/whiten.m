function y = whiten(self, x, varargin)
% Apply this whitener to the given data
%   y = whiten(self, x, ...)
%
% Returns:
%   y       [T_out x C] whitened data (or [T_out x C x N])
% Required arguments:
%   x       [T x C] input data (or [T x C x N])
% Optional parameters (key/value pairs) [default]:
%   bounds  Boundary condition {trunc,center,keep}      ['trunc']
%
% To understand the <bounds> param, consider the following case:
%   wh_filt = [0 0 1 0]' (W=4), delay = 2, x = [1 2 3 4 5]' (T=5)
%     Option   Length   Output             Notes
%     trunc    T-W+1    [3 4]              excludes all overlap with boundaries
%     center       T    [1 2 3 4 5]        depends on the "delay" property
%     keep     T+W-1    [0 0 1 2 3 4 5 0]  includes all possibly-nonzero output

% Optional parameters
ip = inputParser();
ip.addParameter('bounds', 'trunc', @ischar);
ip.parse( varargin{:} );
prm = ip.Results;

% Get the convolver (using the cache if available)
if isempty(self.convolver)
    conv = self.toConv();
    self.convolver = conv;
else
    conv = self.convolver;
end

% Perform the convolution
[T,C,N] = size(x); %#ok<ASGLU>
if (N == 1)
    y = conv.conv(x);
else
    y = conv.conv_batch(x);
end

% Truncate as desired
switch (prm.bounds)
    case 'trunc'
        y = y(self.W:T, :, :);
    case 'center'
        y = y(self.delay+(1:T), :, :);
    case 'keep'
        % Nothing to do
    otherwise
        error(self.errid_arg, 'Unsupported bounds option "%s"',prm.bounds);
end

end
