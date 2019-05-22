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
    function obj = DataSinkInt(sink, datatype)
        % Construct a new DataSinkInt from an underlying sink
        %   obj = DataSinkInt(sink, datatype)
        %
        % Required arguments:
        %   sink      DataSink object to use as the underlying data store
        %   datatype  Integer datatype string (e.g. 'uint8', 'int32', etc)
        obj = obj@spkdec.DataSink(sink.shape, sink.len);
        obj.sink = sink;
        obj.datatype = datatype;
        obj.data_min = intmin(datatype);
        obj.data_max = intmax(datatype);
    end
end

methods (Access=protected)
    % DataSinkInt implementation of the append_internal() method
    function append_internal(self, data)
        assert(all(data >= self.data_min,'all'), self.errid_bounds, ...
            'Integer underflow detected (data < intmin(self.datatype)');
        assert(all(data <= self.data_max,'all'), self.errid_bounds, ...
            'Integer overflow detected (data > intmax(self.datatype))');
        self.sink.append( cast(data,self.datatype) );
        self.len = self.sink.len;
    end
end

end
