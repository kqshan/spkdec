% DataSink subclass that scales/transposes another underlying DataSink
%
% DataSinkScaled properties (read-only)
% Dimensions
%   ext_dim   - Extensible dimension (dimension along which data is appended)
%   shape     - Data dimensions, with Inf for the extensible dimension
%   len       - Current length (# of samples) along the extensible dimension
% Underlying data
%   sink      - Underlying DataSink that actually stores the data
%   scaling   - Scaling factor (or function handle) to apply when writing data
%   perm      - Dimension permutation (e.g. transposition) to apply when writing
%
% DataSinkScaled methods:
%   DataSinkScaled - Constructor
%   hasShape  - Return whether this data sink has the desired shape
%   append    - Write additional data to the end of this data sink

classdef DataSinkScaled < spkdec.DataSink

% ---------------------------     Properties     -------------------------------

properties (SetAccess=protected)
    % Underlying DataSink that actually stores the data
    sink
    
    % Scaling factor (or function handle) to apply when writing data
    % 
    % This can be a numeric scalar, in which case append() writes scaling*data,
    % or a function handle, in which case append() writes scaling(data). This
    % latter form allows you to specify additional type conversion etc. Scaling
    % is applied before permutation.
    scaling

    % Dimension permutation (e.g. transposition) to apply when writing data
    %
    % Specifically, self.append(X) is implemented by:
    %   self.sink.append( permute(X,self.perm) )
    % To transpose a 2-D array, set perm = [2 1]. If this object receives data
    % with dimensions [A x B x C] but the underlying data store is [B x C x A],
    % then set perm = [2 3 1]. Permutation is performed after scaling.
    perm
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = DataSinkScaled(sink, varargin)
        % Construct a new DataSinkScaled from an underlying sink
        %   obj = DataSinkScaled(sink, ...)
        %
        % Required arguments:
        %   sink      DataSink object to use as the underlying data store
        % Optional parameters (key/value pairs) [default]:
        %   scaling   Scaling factor/function to apply      [ 1 ]
        %   perm      Dimension permutation to apply        [1:ndims(src)]
        nDim = length(sink.shape);
        ip = inputParser();
        ip.addParameter('scaling', 1, @isscalar);
        ip.addParameter('perm', 1:nDim, @(x) numel(x)==nDim);
        ip.parse( varargin{:} );
        prm = ip.Results;
        % Figure out the dimensions
        perm = prm.perm(:)';
        dims = zeros(1,nDim);
        dims(perm) = sink.shape;
        % Call the superclass constructor
        obj = obj@spkdec.DataSink(dims, sink.len);
        % Assign values to this object
        obj.sink = sink;
        obj.scaling = prm.scaling;
        obj.perm = perm;
    end
end

methods (Access=protected)
    % DataSinkScaled implementation of the append_internal() method
    function append_internal(self, data)
        % Scale
        if isnumeric(self.scaling)
            data = self.scaling * data;
        else
            data = self.scaling(data);
        end
        % Permute
        if ~issorted(self.perm)
            data = permute(data, self.perm);
        end
        % Append to the underlying sink
        self.sink.append(data);
        self.len = self.sink.len;
    end
end

end
