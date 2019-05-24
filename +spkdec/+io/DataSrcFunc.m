% DataSrc subclass that calls a function to construct an underlying DataSrc
%
% DataSrcFunc properties (read-only):
% Dimensions
%   ext_dim   - Extensible dimension (dimension along which data is read)
%   shape     - Data dimensions, with Inf for the extensible dimension
%   len       - Total length (# of samples) along the extensible dimension
% Underlying data
%   func      - Function handle that constructs an underlying DataSrc
%
% DataSrcFunc methods:
%   DataSrcFunc - Constructor
%   hasShape  - Return whether this data source has the desired shape
%   read      - Read the requested segment of data
%   readNext  - Read the next segment of data
%   readRand  - Read some randomly-selected batches of data
%   planRand  - Plan a set of reads for random batches of data
%
% What is the point of this class? It's a workaround for an issue I'm having
% where I can't seem to open more than 44 DataSrc objects at a time.

classdef DataSrcFunc < spkdec.DataSrc


% ---------------------------     Properties     -------------------------------

properties (SetAccess=protected)
    % Function handle that constructs an underlying DataSrc
    %
    % On each read, this DataSrcFunc object will call this function to construct
    % the underlying DataSrc, perform the read on that DataSrc, and then delete
    % that DataSrc.
    func
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = DataSrcFunc(func)
        % DataSrcFunc constructor
        %   obj = DataSrcFunc(func)
        %
        % Required arguments:
        %   func      Function handle that constructs a DataSrc
        assert(isa(func,'function_handle'));
        % Construct the DataSrc so we can get the dims etc
        src = func();
        obj = obj@spkdec.DataSrc(src.shape, src.len);
        obj.func = func;
        % Delete the DataSrc
        delete(src);
    end
end

methods (Access=protected)
    % Implementation of the abstract read_internal() method
    function x = read_internal(self, start, count)
        src = self.func();
        x = src.read(start, count);
        delete(src);
    end
end

end
