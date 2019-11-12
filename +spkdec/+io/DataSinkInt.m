% DataSink subclass that converts to an integer datatype and checks for overflow
%
% DataSinkInt properties (read-only)
% Dimensions
%   ext_dim   - Extensible dimension (dimension along which data is appended)
%   shape     - Data dimensions, with Inf for the extensible dimension
%   len       - Current length (# of samples) along the extensible dimension
% Underlying data
%   sink      - Underlying DataSink that actually stores the data
%   datatype  - Datatype string
%   err_mode  - What to do when an integer overflow occurs: {err,warn,none}
%
% DataSinkInt methods:
%   DataSinkInt - Constructor
%   hasShape  - Return whether this data sink has the desired shape
%   append    - Write additional data to the end of this data sink

classdef DataSinkInt < spkdec.DataSink

% ---------------------------     Properties     -------------------------------

properties (SetAccess=protected)
    % Underlying DataSink that actually stores the data
    sink
    
    % Datatype string (e.g. 'uint8', 'int32', etc)
    datatype
    
    % What to do when an integer overflow occurs: {err, warn, none}
    %
    % Potential modes:
    %   err     - Throw an error
    %   warn    - Produce a warning (map out-of-bounds values to intmin/intmax)
    %   none    - Silently map out-of-bounds values to intmin/intmax
    err_mode
end

properties (Access=protected)
    % intmin and intmax for this datatype
    data_min
    data_max
end

properties (Hidden, Constant)
    errid_bounds = 'spkdec:io:DataSinkInt:OutOfBounds';
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = DataSinkInt(sink, datatype, varargin)
        % Construct a new DataSinkInt from an underlying sink
        %   obj = DataSinkInt(sink, datatype, ...)
        %
        % Required arguments:
        %   sink      DataSink object to use as the underlying data store
        %   datatype  Integer datatype string (e.g. 'uint8', 'int32', etc)
        % Optional parameters (key/value pairs) [default]:
        %   err_mode  What to do with integer overflow {[err],warn,none}
        ip = inputParser();
        ip.addParameter('err_mode', 'err', @ischar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        obj = obj@spkdec.DataSink(sink.shape, sink.len);
        obj.sink = sink;
        obj.datatype = datatype;
        obj.data_min = intmin(datatype);
        obj.data_max = intmax(datatype);
        assert(ismember(prm.err_mode, {'err','warn','none'}), ...
            obj.errid_arg, 'Unsupported err_mode "%s"',prm.err_mode);
        obj.err_mode = prm.err_mode;
    end
end

methods (Access=protected)
    % DataSinkInt implementation of the append_internal() method
    function append_internal(self, data)
        if ~strcmp(self.err_mode,'none')
            % Check the bounds
            has_overflow = any(data < self.data_min,'all') ...
                || any(data > self.data_max,'all');
            if ~has_overflow
                % No need to do anything
            elseif strcmp(self.err_mode,'err')
                error(self.errid_bounds, 'Integer overflow detected');
            elseif strcmp(self.err_mode,'warn')
                warning(self.errid_bounds, 'Integer overflow detected');
            end
        end
        % Append the data
        self.sink.append( cast(data,self.datatype) );
        self.len = self.sink.len;
    end
end

end
