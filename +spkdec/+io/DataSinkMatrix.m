% DataSink subclass using a matrix as the underlying data store
%
% DataSinkMatrix properties (read-only):
% Dimensions
%   ext_dim   - Extensible dimension (dimension along which data is appended)
%   shape     - Data dimensions, with Inf for the extensible dimension
%   len       - Current length (# of samples) along the extensible dimension
% Underlying data
%   data      - Underlying data store (matrix or matrix-like object)
%
% DataSinkMatrix methods:
%   DataSinkMatrix - Constructor
%   hasShape  - Return whether this data sink has the desired shape
%   append    - Write additional data to the end of this data sink
%   cast      - Cast the underlying data to a different type
%
% Note that <data> does not necessarily need to be a MATLAB matrix. It can be
% any class with appropriately overloaded size() and cat() methods.

classdef DataSinkMatrix < spkdec.DataSink

% ---------------------------     Properties     -------------------------------

properties (SetAccess=protected)
    % Underlying data store (matrix or matrix-like object)
    %
    % This is a numeric matrix that gets appended to. Alternatively, this can be
    % any class for which the size() and cat() methods have been appropriately
    % overloaded.
    %
    % If <data> is a numeric matrix, then append() will also typecast the given
    % data to match it.
    data
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = DataSinkMatrix(shape, data)
        % DataSinkMatrix constructor
        %   obj = DataSinkMatrix(shape, [data])
        %
        % Required arguments:
        %   shape   Data dimensions (row vector), with Inf for ext. dim
        % Optional arguments [default]:
        %   data    Underlying data store                   [ zeros ]
        obj = obj@spkdec.DataSink(shape);
        % Initialize the data
        if nargin < 2
            shape(obj.ext_dim) = 0;
            data = zeros([shape, 1]); % Trailing dim in case <shape> is scalar
        else
            assert(obj.hasShape(size(data),'strict',false), obj.errid_dim, ...
                'Given data must match the specified shape');
            obj.len = size(data, obj.ext_dim);
        end
        obj.data = data;
    end
    
    function cast(self, varargin)
        % Cast the underlying data to a different datatype
        %   cast(self, class_name)
        %   cast(self, 'like', example)
        narginchk(2,3);
        self.data = cast(self.data, varargin{:});
    end
end

methods (Access=protected)
    % DataSinkMatrix implementation of the append_internal() method
    function append_internal(self, data)
        % Typecast to match
        if isnumeric(self.data)
            data = cast(data, 'like',self.data);
        end
        % Append
        self.data = cat(self.ext_dim, self.data, data);
    end
end

end
