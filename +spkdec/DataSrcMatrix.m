% Subclass of DataSrc that is implemented using a matrix
%
% DataSrcMatrix properties (read-only):
% Dimensions
%   C         - Number of channels
%   T         - Total length (# of samples)
% Underlying data
%   data      - Underlying data store (matrix or matrix-like object)
%   transp    - Data is transposed (i.e. is [C x T] instead of [T x C])
%   scaling   - Factor to apply when reading data
%
% DataSrcMatrix methods:
%   DataSrcMatrix - Constructor
%   read      - Read the requested segment of data
%   readRand  - Read some randomly-selected batches of data
%   copy      - Create a shallow copy of this handle object
%
% Note that <data> does not necessarily need to be a MATLAB matrix. It can be
% any class with appropriately overloaded size() and subsref() methods.

classdef DataSrcMatrix < spkdec.DataSrc


% ------------------------------------------------------------------------------
% ========================     Public properties     ===========================
% ------------------------------------------------------------------------------


properties (Dependent, SetAccess=private)
    C
    T
end
methods
    function val = get.C(self)
        sz = size(self.data); if self.transp, val=sz(1); else, val=sz(2); end
    end
    function val = get.T(self)
        sz = size(self.data); if self.transp, val=sz(2); else, val=sz(1); end
    end
end

properties (SetAccess=protected)
    % Underlying data store (matrix or matrix-like object)
    % This is a [T x C] (or [C x T] if transp==true) numeric matrix that the
    % read() method will access. Alternatively, this can be any class for which
    % the size() and subsref() methods have been overloaded to allow matrix-like
    % access to its data.
    data
    
    % Data is transposed (i.e. is [C x T] instead of [T x C])
    % This flag indicates whether the data store is transposed. Setting
    % <transp>=true will change how <data> is indexed in read(), and will cause
    % the data to be tranposed before being returend by read().
    transp = false;
    
    % Factor to apply when reading the data
    % This can be a numeric scalar, in which case read() returns scaling*data,
    % or a function handle, in which case read() returns scaling(data). This
    % latter form allows you to specify additional type conversion etc. If
    % <transp>==true, then <scaling> is applied after transposition.
    scaling = 1;
end


% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------


methods
    function obj = DataSrcMatrix(data, varargin)
        % DataSrcMatrix constructor
        %   obj = DataSrcMatrix(data, ...)
        %
        % Required arguments:
        %   data      [T x C] or [C x T] matrix or matrix-like object
        % Optional parameters (key/value pairs) [default]:
        %   transp    Data is transposed (i.e. [C x T])         [ false ]
        %   scaling   Factor to apply when reading the data     [ 1 ]
        ip = inputParser();
        ip.addParameter('transp', false, @isscalar);
        ip.addParameter('scaling', 1, @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        obj.data = data;
        obj.transp = prm.transp;
        obj.scaling = prm.scaling;
    end
end


% ------------------------------------------------------------------------------
% ============================      Protected      =============================
% ------------------------------------------------------------------------------


methods (Access=protected)
    % Implementation of the abstract read_internal() method
    function x = read_internal(self, start, len)
        % DataSrcMatrix implementation of the read_internal() method
        %   x = read_internal(self, start, len)
        %
        % Returns:
        %   x       [len x C] data
        % Required arguments:
        %   start   Index of the first data sample to read
        %   len     Length of data to read (#samples)
        read_range = start:start+len-1;
        if self.transp
            x = self.data(:,read_range).';
        else
            x = self.data(read_range,:);
        end
        % Apply the scaling factor
        if isnumeric(self.scaling)
            x = self.scaling * x;
        else
            x = self.scaling(x);
        end
    end
end

end