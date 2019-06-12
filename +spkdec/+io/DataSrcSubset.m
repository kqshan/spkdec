% DataSrc subclass that reads a subset of another underlying DataSrc
%
% DataSrcSubset properties (read-only):
% Dimensions
%   ext_dim   - Extensible dimension (dimension along which data is read)
%   shape     - Data dimensions, with Inf for the extensible dimension
%   len       - Total length (# of samples) along the extensible dimension
% Underlying data
%   src       - Underlying DataSrc that actually stores the data
%   offset    - Offset relative to the underlying DataSrc
%
% DataSrcSubset methods:
%   DataSrcSubset - Constructor
%   hasShape  - Return whether this data source has the desired shape
%   read      - Read the requested segment of data
%   readNext  - Read the next segment of data
%   readRand  - Read some randomly-selected batches of data
%   planRand  - Plan a set of reads for random batches of data


classdef DataSrcSubset < spkdec.DataSrc

% ---------------------------     Properties     -------------------------------

properties (SetAccess=protected)
    % Underlying DataSrc that actually stores the data
    src
    
    % Offset relative to the underlying DataSrc
    offset
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = DataSrcSubset(src, offset, len)
        % Construct a new DataSrcSubset from an underlying source
        %   obj = DataSrcSubset(src, offset, len)
        %
        % Required arguments:
        %   src       DataSrc object to use as the underlying data store
        %   offset    Offset relative to the underlying DataSrc
        %   len       Total length of this subset
        assert(offset >= 0, spkdec.DataSrc.errid_arg, ...
            'offset must be nonnegative');
        assert(offset+len <= src.len, spkdec.DataSrc.errid_arg, ...
            'Selected subset cannot extend beyond underlying DataSrc');
        obj = obj@spkdec.DataSrc(src.shape, len);
        obj.src = src;
        obj.offset = offset;
    end
end

methods (Access=protected)
    % DataSrcSubset implementation of the read_internal() method
    function x = read_internal(self, start, count)
        x = self.src.read(start+self.offset, count);
    end
end

end
