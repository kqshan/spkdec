% DataSrc subclass that calls another function to read the requested data
%
% DataSrcCallback properties (read-only):
% Dimensions
%   ext_dim   - Extensible dimension (dimension along which data is read)
%   shape     - Data dimensions, with Inf for the extensible dimension
%   len       - Total length (# of samples) along the extensible dimension
% Underlying data
%   callback  - Function handle for reading data
%
% DataSrcCallback methods:
%   DataSrcCallback - Constructor
%   hasShape  - Return whether this data source has the desired shape
%   read      - Read the requested segment of data
%   readNext  - Read the next segment of data
%   readRand  - Read some randomly-selected batches of data
%   planRand  - Plan a set of reads for random batches of data

classdef DataSrcCallback < spkdec.DataSrc

% ---------------------------     Properties     -------------------------------

properties (SetAccess=protected)
    % Function handle that reads the requested segment of data
    %
    % This function will be called whenever this DataSrcCallback object is
    % requested to read something. It is expected to perform the following:
    %   x = callback(start, count)
    % Returns:
    %   x       Data array (size(x) == self.shape, except for ext_dim)
    % Required arguments:
    %   start   Index of first data sample to read (starts at 1)
    %   count   Length of data to read (#samples)
    callback
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = DataSrcCallback(callback, shape, len)
        % DataSrcCallback constructor
        %   obj = DataSrcCallback(callback, shape, len)
        %
        % Required arguments:
        %   callback    Callback function to implement the read
        %   shape       Data dimensions (row vec), with Inf for extensible dim
        %   len         Length along the extensible dimension
        obj = obj@spkdec.DataSrc(shape, len);
        obj.callback = callback;
    end
end

methods (Access=protected)
    % Implementation of the abstract read_internal() method
    function x = read_internal(self, start, count)
        x = self.callback(start, count);
    end
end

end
