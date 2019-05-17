% DataSrc subclass that scales/transposes another underlying DataSrc
%
% DataSrcScaled properties (read-only)
% Dimensions
%   ext_dim   - Extensible dimension (dimension along which data is read)
%   shape     - Data dimensions, with Inf for the extensible dimension
%   len       - Total length (# of samples) along the extensible dimension
% Underlying data
%   src       - Underlying DataSrc that actually stores the data
%   perm      - Dimension permutation (e.g. transposition) to apply when reading
%   scaling   - Scaling factor (or function handle) to apply when reading data
%
% DataSrcScaled methods:
%   DataSrcScaled - Constructor
%   hasShape  - Return whether this data source has the desired shape
%   read      - Read the requested segment of data
%   readNext  - Read the next segment of data
%   readRand  - Read some randomly-selected batches of data
%   planRand  - Plan a set of reads for random batches of data


classdef DataSrcScaled < spkdec.DataSrc

% ---------------------------     Properties     -------------------------------

properties (SetAccess=protected)
    % Underlying DataSrc that actually stores the data
    src
    
    % Dimension permutation (e.g. transposition) to apply when reading data
    %
    % Specifically, self.read(X) is implemented by:
    %   permute(self.src.read(...), self.perm)
    % To transpose a 2-D array, set perm = [2 1]. If this object contains data
    % with dimensions [A x B x C] but you wish to read it as [B x C x A], then
    % set perm = [2 3 1]. Permutation is performed before scaling.
    perm

    % Scaling factor (or function handle) to apply when reading data
    % 
    % This can be a numeric scalar, in which case read() returns scaling*data,
    % or a function handle, in which case read() returns scaling(data). This
    % latter form allows you to specify additional type conversion etc. Scaling
    % is applied after permutation.
    scaling
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = DataSrcScaled(src, varargin)
        % Construct a new DataSrcScaled from an underlying source
        %   obj = DataSrcScaled(src, ...)
        %
        % Required arguments:
        %   src       DataSrc object to use as the underlying data store
        % Optional parameters (key/value pairs) [default]:
        %   scaling   Scaling factor/function to apply      [ 1 ]
        %   perm      Dimension permutation to apply        [1:ndims(src)]
        nDim = length(src.shape);
        ip = inputParser();
        ip.addParameter('scaling', 1, @isscalar);
        ip.addParameter('perm', 1:nDim, @(x) numel(x)==nDim);
        ip.parse( varargin{:} );
        prm = ip.Results;
        % Figure out the dimensions
        perm = prm.perm(:)';
        dims = zeros(1,nDim);
        dims(perm) = src.shape;
        % Call the superclass constructor
        obj = obj@spkdec.DataSrc(dims, src.len);
        % Assign values to this object
        obj.src = src;
        obj.perm = perm;
        obj.scaling = prm.scaling;
    end
end

methods (Access=protected)
    % DataSrcScaled implementation of the read_internal() method
    function x = read_internal(self, start, count)
        x = self.src.read(start, count);    % Read
        x = permute(x, self.perm);          % Permute
        if isnumeric(self.scaling)          % Scale
            x = self.scaling * x;
        else
            x = self.scaling(x);
        end
    end
end

end
