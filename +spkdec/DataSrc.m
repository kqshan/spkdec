% Abstract interface defining a data source with one extensible dimension
%
% DataSrc properties (read-only):
%   ext_dim   - Extensible dimension (dimension along which data is read)
%   shape     - Data dimensions, with Inf for the extensible dimension
%   len       - Total length (# of samples) along the extensible dimension
%
% DataSrc methods:
%   hasShape  - Return whether this data source matches the given dimensions
%   read      - Read the requested segment of data
% Streaming-style access
%   readNext  - Read the next segment of data
% Random sampling
%   readRand  - Read some randomly-selected batches of data
%   planRand  - Plan a set of reads for random batches of data

% Subclasses must implement the protected read_internal() method. If a subclass
% is only capable of streaming-style access, it still must implement
% read_internal(), if only to throw an error, and then it can overload hasNext()
% and readNext() to perform the actual read.

classdef (Abstract) DataSrc < spkdec.io.DataInterface

% ---------------------------     Properties     -------------------------------

properties (Access=protected)
    % Current position (index of last sample read). Used to implement readNext()
    pos
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = DataSrc(shape, len, pos)
        % DataSrc constructor
        %   obj = DataSrc(shape, len, [pos])
        %
        % Required arguments:
        %   shape   Data dimensions (row vector), with Inf for extensible dim
        %   len     Total length along the extensible dimension
        % Optional arguments [default]:
        %   pos     Current position for readNext()             [ 0 ]
        if nargin < 3, pos = 0; end
        obj = obj@spkdec.io.DataInterface(shape, len);
        obj.pos = pos;
    end
    
    % Primary read function --------------------------------------------
    
    function x = read(self, start, count)
        % Read the requested segment of data
        %   x = read(self, start, count)
        %
        % Returns:
        %   x       Data array (size(x) == self.shape, except for ext_dim)
        % Required arguments:
        %   start   Index of first data sample to read
        %   count   Length of data to read (#samples)
        %
        % This updates the state of the readNext() method, even if count==0,
        % so it can be used to seek to a position in the underlying data.
        assert(start >= 1, self.errid_dim, 'Start index must be >= 1');
        assert(start+count-1 <= self.len, self.errid_dim, ...
            'End index exceeds the overall data length');
        x = self.read_internal(start, count);
        self.pos = start-1 + count;
    end
    
    % Streaming read ---------------------------------------------------
    
    function [x, islast] = readNext(self, count)
        % Read the next segment of data (for streaming-style access)
        %   [x, islast] = readNext(self, count)
        %
        % Returns:
        %   x       Data array (size(x)==self.shape, except for ext_dim)
        %   islast  Whether this was the last segment of data in this source
        % Required arguments:
        %   count   Maximum length of data to read
        remaining = self.len - self.pos;
        islast = (remaining <= count);
        count = min(remaining, count);
        x = self.read_internal(self.pos+1, count);
        self.pos = self.pos + count;
    end
    
    % Random sampling --------------------------------------------------
    
    function x = readRand(self, varargin)
        % Read some randomly-selected batches of data
        %   x = readRand(self, ...)
        %
        % Returns:
        %   x           [... x n_batch] data batches
        % Optional parameters (key/value pairs) [default]:
        %   batch_size  Size (#samples) of each batch           [ 16k ]
        %   n_batch     Number of batches to select             [ 64 ]
        %   overlap     Allowable overlap between batches       [ 0 ]
        %   strict      Force x to meet given dimensions        [ false ]
        %
        % If strict==false and the total data length <= n_batch*batch_size,
        % then the entire dataset will be returned as a single batch.
        %
        % This does not alter the state of the readNext() method.
        [starts, count] = self.planRand(varargin{:});
        x = arrayfun(@(start) {self.read_internal(start,count)}, starts);
        x = cat(length(self.shape)+1, x{:});
    end
    
    function [starts, count] = planRand(self, varargin)
        % Randomly select a set of data batches to read
        %   [starts, count] = planRand(self, varargin)
        %
        % Returns:
        %   starts      [n_batch x 1] index (1..len) of first sample in batch
        %   count       Batch length (same as batch_size)
        % Optional parameters (key/value pairs) [default]:
        %   batch_size  Size (#samples) of each batch           [ 16k ]
        %   n_batch     Number of batches to select             [ 64 ]
        %   overlap     Allowable overlap between batches       [ 0 ]
        %   strict      Force x to meet given dimensions        [ false ]
        %
        % If strict==false and the total data length <= n_batch*batch_size,
        % then the entire dataset will be returned as a single batch, i.e.
        % starts=1 and count=len.
        ip = inputParser();
        ip.addParameter('batch_size', 16*1024, @isscalar);
        ip.addParameter('n_batch', 64, @isscalar);
        ip.addParameter('overlap', 0, @isscalar);
        ip.addParameter('strict', false, @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        % Collect some dimensions
        count = prm.batch_size; B = prm.n_batch; ovlp = prm.overlap;
        stride = count - ovlp;
        T_all = self.len;
        % Plan out the batches
        total_gap = T_all - B*stride - ovlp;
        if total_gap < 0
            % There isn't enough data for the desired batches
            assert(~prm.strict, self.errid_dim, ['Data is too short to ' ...
                'satisfy the given batchSize, nBatch, overlap specs']);
            offsets = 0;
            count = T_all;
        else
            % Randomly select the batches
            gaps = diff([0; sort(randi([0, total_gap], [B 1]))]);
            offsets = cumsum(stride+gaps) - stride;
        end
        starts = offsets + 1;
    end
end

methods (Access=protected, Abstract)
    % Protected method to read from the underlying data store
    %   x = read_internal(self, start, count)
    %
    % This method must be implemented by subclasses
    x = read_internal(self, start, count);
end

end
