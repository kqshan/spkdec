function tf = has_gpu(varargin)
% Return whether this machine has a GPU or not
%   tf = has_gpu(...)
%
% Returns:
%   tf          Logical scalar indicating whether a GPU is available
% Optional parameters (key/value pairs) [default]:
%   override    Value to override this test with        [ none ]
%
% This stores the result of this test in a persistent variable. If you call this
% with <override> set, then subsequent calls to this function will return that
% value. For example:
% >> spkdec.Math.has_gpu()
% ans =
%    1
%
% >> spkdec.Math.has_gpu('override',false)
% ans =
%    0
% 
% >> spkdec.Math.has_gpu()
% ans =
%    0

persistent has_gpu_val

ip = inputParser();
ip.addParameter('override', [], @(x) isempty(x) || isscalar(x));
ip.parse( varargin{:} );
prm = ip.Results;

% Use the override first
if ~isempty(prm.override)
    tf = prm.override;
    has_gpu_val = tf;
    return
end

% Then try the persistent variable
if ~isempty(has_gpu_val)
    tf = has_gpu_val;
    return
end

% Then try to construct a gpuArray
try
    x = gpuArray(); %#ok<NASGU>
    tf = true;
catch
    tf = false;
end
has_gpu_val = tf;

end
