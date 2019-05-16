% Abstract interface defining a data sink with one extensible dimension
%
% DataSink properties (read-only):
%   ext_dim   - Extensible dimension (dimension along which data is appended)
%   shape     - Data dimensions, with Inf for the extensible dimension
%   len       - Current length (# of samples) along the extensible dimension
%
% DataSink methods:
%   hasShape  - Return whether this data sink matches the given dimensions
%   append    - Write additional data to the end of this data sink

% Subclasses must implement the protected append_internal() method.

classdef (Abstract) DataSink < spkdec.io.DataInterface

% -----------------------------     Methods     --------------------------------

methods
    function append(self, data)
        % Append the given data to the end of this data sink
        %   append(self, data)
        %
        % Required arguments:
        %   data    Data ta append to this data sink. size(data) must match
        %           self.shape (except for the extensible dimension)
        assert(self.hasShape(size(data), 'strict',false), ...
            'spkdec:DataSink:DimMismatch', ...
            'Given data must match the shape of this DataSink');
        self.append_internal(data);
    end
end

% ---------  Protected methods for subclasses to implement/overload  -----------

methods (Access=protected, Abstract)
    % Protected method to append the given data to the underlying data store
    %   append_internal(self, data)
    %
    % This abstract method must be implemented by subclasses
    append_internal(self, data);
end

end
