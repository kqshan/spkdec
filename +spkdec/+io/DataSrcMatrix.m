% Subclass of DataSrc that is implemented using a matrix
%
% DataSrcMatrix properties (read-only):
% Dimensions
%   ext_dim   - Extensible dimension (dimension along which data is read)
%   shape     - Data dimensions, with Inf for the extensible dimension
%   len       - Total length (# of samples) along the extensible dimension
% Underlying data
%   data      - Underlying data store (matrix or matrix-like object)
%
% DataSrcMatrix methods:
%   DataSrcMatrix - Constructor
%   hasShape  - Return whether this data source has the desired shape
%   read      - Read the requested segment of data
%   readNext  - Read the next segment of data
%   readRand  - Read some randomly-selected batches of data
%   planRand  - Plan a set of reads for random batches of data
%
% Note that <data> does not necessarily need to be a MATLAB matrix. It can be
% any class with appropriately overloaded size() and subsref() methods.

classdef DataSrcMatrix < spkdec.DataSrc


% ---------------------------     Properties     -------------------------------

properties (SetAccess=protected)
    % Underlying data store (matrix or matrix-like object)
    %
    % This is a numeric matrix that will get read from. Alternatively, this can
    % be any class for which the size() and subsref() methods have been
    % appropriately overloaded.
    data
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = DataSrcMatrix(data, ext_dim, ndims)
        % DataSrcMatrix constructor
        %   obj = DataSrcMatrix(data, ext_dim, [ndims])
        %
        % Required arguments:
        %   data      Underlying data store
        %   ext_dim   Extensible dimension (dim. along which data will be read)
        % Optional arguments [default]:
        %   ndims     Number of dimensions in data store            [ auto ]
        errid_arg = spkdec.DataSrc.errid_arg;
        % Determining the number of dimensions is actually kinda tricky
        shape = size(data);
        ndims_data = length(shape);
        if nargin < 3
            ndims = ndims_data;
            % Special case for column vectors
            if ndims==2 && shape(2)==1 && ext_dim==1
                ndims = 1;
                shape = shape(1);
            end
        elseif ndims > ndims_data
            % Add trailing singleton dimensions if necessary
            shape = [shape, ones(1,ndims-ndims_data)];
        elseif ndims < ndims_data
            % This is generally not okay, but column vectors are a special case
            if (ndims==1) && (ndims_data==2) && (shape(2)==1)
                shape = shape(1);
            else
                error(errid, 'ndims must be >= length(size(data))');
            end
        end
        assert(ext_dim <= ndims, errid_arg, 'ext_dim must be <= ndims');
        % Call the superclass constructor
        len = shape(ext_dim);
        shape(ext_dim) = Inf;
        obj = obj@spkdec.DataSrc(shape, len);
        obj.data = data;
    end
end

methods (Access=protected)
    % Implementation of the abstract read_internal() method
    function x = read_internal(self, start, count)
        % Indicate that we want (:,:,start:start-1+count,:)
        N = length(self.shape);
        S = struct('type','()');
        S.subs = repmat({':'}, [1 N]);
        S.subs{self.ext_dim} = start:start-1+count;
        % Perform the subsref operation
        x = subsref(self.data, S);
    end
end

end
