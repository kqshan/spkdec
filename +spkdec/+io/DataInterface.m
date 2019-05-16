% Data with a single extensible dimension (superclass of DataSrc and DataSink)
%
% DataInterface properties (read-only):
%   ext_dim   - Extensible dimension (dimension along which data is unlimited)
%   shape     - Data dimensions, with Inf for the extensible dimension
%   len       - Current length (# of samples) along the extensible dimension
%
% DataInterface methods:
%   DataInterface - Constructor
%   hasShape      - Return whether this data has the desired shape

classdef DataInterface < handle

% ---------------------------     Properties     -------------------------------

properties (SetAccess=private, Dependent)
    % Extensible dimension (dimension along which data is unlimited)
    %
    % This is the dimension along which the data will be appended (for a
    % DataSink) or along which it will be read in segments (for a DataSrc)
    ext_dim
end
methods
    function val = get.ext_dim(self), val = find(isinf(self.shape)); end
end

properties (SetAccess=protected)
    % Data dimensions (row vector), with Inf for the extensible dimension
    shape
    
    % Current length (# of samples) along the extensible dimension
    len
end

properties (Constant, Hidden)
    errid_dim = 'spkdec:DataInterface:DimMismatch';
    errid_arg = 'spkdec:DataInterface:BadArg';
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = DataInterface(shape, len)
        % Construct a new DataInterface with specified shape and length
        %   obj = DataInterface(shape, [len])
        %
        % Required arguments:
        %   shape   Data dimensions (row vector), with Inf for extensible dim
        % Optional arguments [default]:
        %   len     Current length along the extensible dimension       [ 0 ]
        if nargin < 2, len = 0; end
        assert(isvector(shape) && sum(isinf(shape))==1, obj.errid_arg, ...
            '<shape> must be a vector with a single Inf dimension');
        obj.shape = shape(:)'; % Make sure it's a row vector
        obj.len = len;
    end
    
    function tf = hasShape(self, shape, varargin)
        % Return whether this data object has the desired shape
        %   tf = hasShape(self, shape, ...)
        %
        % Returns:
        %   tf      Whether this data object has the desired shape
        % Required arguments:
        %   shape   Desired dimensions (row vector), with Inf for extensible dim
        % Optional parameters (key/value pairs) [default]:
        %   strict  Require strict matching of the dimensions   [ true ]
        %
        % If strict==true, then this requires an exact match, i.e.
        %   tf = isequal(self.shape,shape)
        % If strict==false, then shape(self.ext_dim) can be any value, and this
        % will append or truncate trailing singleton dimensions as necessary,
        % which is useful if trying to match with the output of size().
        shape = shape(:)'; % Make sure it's a row vector
        ip = inputParser();
        ip.addParameter('strict', true, @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        if prm.strict
            % Compare them directly
            tf = isequal(self.shape, shape);
        else
            % Append or truncate trailing singleton dimensions
            N = length(shape);
            self_N = length(self.shape);
            if N > self_N
                % Given shape has more dimensions than us
                if any(shape(self_N+1:end) ~= 1)
                    tf = false;
                    return
                end
                shape = shape(1:self_N);
            elseif N < self_N
                shape = [shape, ones(1,self_N-N)];
            end
            % Compare them
            tf = all((shape==self.shape) | isinf(self.shape));
        end
    end
end

end
