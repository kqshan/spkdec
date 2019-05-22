% DataSink subclass using an HDF5 as the underlying data store
%
% DataSinkH5 properties (read-only):
% Dimensions
%   ext_dim   - Extensible dimension (dimension along which data is appended)
%   shape     - Data dimensions, with Inf for the extensible dimension
%   len       - Current length (# of samples) along the extensible dimension
% Underlying data
%   filename  - HDF5 file name
%   dsname    - Dataset name within the HDF5 file
%
% DataSinkH5 methods:
%   DataSinkH5 - Constructor
%   hasShape  - Return whether this data sink has the desired shape
%   append    - Write additional data to the end of this data sink

classdef DataSinkH5 < spkdec.DataSink

% ---------------------------     Properties     -------------------------------

properties (SetAccess=protected)
    % HDF5 file name
    filename
    
    % Dataset name within the HDF5 file
    dsname
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = DataSinkH5(shape, filename, varargin)
        % DataSinkH5 constructor
        %   obj = DataSinkH5(shape, filename, ...)
        %
        % Required arguments:
        %   shape       Data dimensions (row vector), with Inf for ext. dim
        %   filename    HDF5 filename
        % Optional parameters (key/value pairs) [default]:
        %   dsname      Dataset name within the HDF5 file       ['/dataset']
        %   old_file_ok Okay if the file already exists         [ true ]
        %   old_ds_ok   Okay if the dataset already exists      [ false ]
        %   chunk_size  HDF5 chunk size in the extensible dim   [ 4096 ]
        %   ...         Additional args will be forwarded to h5create()
        errid_arg = spkdec.DataSink.errid_arg;
        % Parse the optional parameters
        ip = inputParser();
        ip.KeepUnmatched = true; ip.PartialMatching = false;
        ip.addParameter('dsname', '/dataset', @ischar);
        ip.addParameter('old_file_ok', true, @isscalar);
        ip.addParameter('old_ds_ok', false, @isscalar);
        ip.addParameter('chunk_size', 4096, @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        addl_args = ip.Unmatched;
        % See if the file and/or dataset already exists
        dsname = prm.dsname;
        if exist(filename,'file')
            assert(prm.old_file_ok, errid_arg, ...
                'File "%s" already exists; aborting', filename);
            ds_info = read_ds_info(filename, dsname);
            assert(isempty(ds_info) || prm.old_ds_ok, errid_arg, ...
                '%s already contains dataset "%s"; aborting', filename, dsname);
        else
            ds_info = [];
        end
        % Call the superclass constructor
        obj = obj@spkdec.DataSink(shape);
        obj.filename = filename;
        obj.dsname = dsname;
        % Create a new dataset or append to an existing one
        ext_dim = obj.ext_dim;
        if isempty(ds_info)
            chunk_sz = shape;
            chunk_sz(ext_dim) = prm.chunk_size;
            h5create(filename, dsname, shape, 'ChunkSize',chunk_sz, addl_args);
        else
            assert(isinf(ds_info.Dataspace.MaxSize(ext_dim)), errid_arg, ...
                'Existing dataset must be unlimited size along dim %d',ext_dim);
            obj.len = ds_info.Dataspace.Size(ext_dim);
        end
    end
end

methods (Access=protected)
    % DataSinkH5 implementation of this abstract method
    function append_internal(self, data)
        N = length(self.shape);
        d = self.ext_dim;
        % Set the data start point
        start = ones(1,N);
        start(d) = self.len + 1;
        % Add/remove trailing singleton dimensions on the data count
        count = size(data);
        M = length(count);
        if (M < N)
            count = [count, ones(1,N-M)];
        elseif (M > N)
            count(N) = prod(count(N:end));
            count = count(1:N);
        end
        T = count(d);
        % Write the data
        h5write(self.filename, self.dsname, data, start, count);
        % Increment the length
        self.len = self.len + T;
    end
end

end


% -----------------------     Helper functions     -----------------------------


function info = read_ds_info(filename, dsname)
% Return the h5info struct for the selected HDF5 dataset, or [] if not found
%   info = read_ds_info(filename, dsname)
try
    info = h5info(filename, dsname);
catch mexc
    if strcmp(mexc.identifier,'MATLAB:imagesci:h5info:unableToFind')
        info = [];
    else
        rethrow(mexc);
    end
end
end
