% DataSink subclass using a cell array of matrices as the underyling data store
%
% DataSinkCell properties (read-only):
% Dimensions
%   ext_dim   - Extensible dimension (dimension along which data is appended)
%   shape     - Data dimensions, with Inf for the extensible dimension
%   len       - Current length (# of samples) along the extensible dimension
% Underlying data
%   data_arr  - Cell array of matrices that act as the underlying data store
%
% DataSinkCell methods:
%   DataSinkCell - Constructor
%   hasShape  - Return whether this data sink has the desired shape
%   append    - Write additional data to the end of this data sink
%   get_data  - Return the data as a single concatenated matrix

classdef DataSinkCell < spkdec.DataSink

% ---------------------------     Properties     -------------------------------

properties (SetAccess=protected)
    % Cell array of matrices that act as the underlying data store
    %
    % Each time that append() is called, this appends a new cell to this array.
    data_arr
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = DataSinkCell(shape)
        % DataSinkCell constructor
        %   obj = DataSinkCell(shape)
        %
        % Required arguments:
        %   shape   Data dimensions (row vector), with Inf for ext. dim
        obj = obj@spkdec.DataSink(shape);
        obj.data_arr = {};
    end
    
    function data = get_data(self)
        % Return the data as a single concatenated matrix
        %   data = get_data(self)
        %
        % Returns:
        %   data    MATLAB array containing all the data in self.data_arr
        %           concatenated together along the extensible dimension
        data = cat(self.ext_dim, self.data_arr{:});
    end
end

methods (Access=protected)
    % DataSinkCell implementation of the append_internal() method
    function append_internal(self, data)
        self.data_arr{end+1,1} = data;
    end
end

end
