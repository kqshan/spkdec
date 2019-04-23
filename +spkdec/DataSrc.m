% Abstract class that defines an interface to a data source
%
% DataSrc properties:
%   C         - Number of channels
%   T         - Total length (# of samples)
%
% DataSrc methods:
%   read      - Read the requested segment of data
%   readRand  - Read some randomly-selected batches of data
%   planRand  - Plan a set of reads for random batches of data

classdef (Abstract) DataSrc < matlab.mixin.Copyable

% ---------------------------     Properties     -------------------------------

properties (Abstract, SetAccess=private)
    % Number of channels in this data source
    C
    
    % Total length (# of samples) in this data source
    T
end

properties (Constant, Access=private)
    errid_dim = 'spkdec:DataSrc:DimMismatch';
end

% -----------------------------     Methods     --------------------------------

methods (Access=protected, Abstract)
    % Subclasses must implement this
    x = read_internal(self, start, len);
end

methods
    function x = read(self, start, len, varargin)
        % Read the requested segment of data
        %   x = read(self, [start, len, ]...)
        %
        % Returns:
        %   x         [len x C] data
        % Optional arguments [default]:
        %   start     Index of first data sample to read        [ 1 ]
        %   len       Length of data to read (#samples)         [ T ]
        % Optional parameters (key/value pairs) [default]:
        %   strict    Force x to meet the given dimensions      [ false ]
        %
        % If strict==false and the total data length T < (start-1)+len, then
        % only the remaining data (possibly empty) will be read and returned
        T_ = self.T;
        if nargin < 2, start = 1; end
        assert(start >= 1, self.errid_dim, 'Start index must be >= 1');
        if nargin < 3, len = T_; end
        ip = inputParser();
        ip.addParameter('strict', false, @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        % Determine the actual range to read
        last = (start-1) + len;
        if last > T_
            assert(~prm.strict, self.errid_dim, ['End index (%d) ' ...
                'exceeds the overall data length (%d)'], last, T_);
            len = T_ - (start-1); % last = T_; len = last - (start-1)
        end
        % Special case for empty reads
        if len <= 0
            x = zeros(0,self.C);
            return
        end
        % Call the internal read() method
        x = self.read_internal(start, len);
    end
    
    
    function x = readRand(self, varargin)
        % Read some randomly-selected batches of data
        %   x = readRand(self, ...)
        %
        % Returns:
        %   x           [batch_size x C x n_batch] data batches
        % Optional parameters (key/value pairs) [default]:
        %   batch_size  Size (#samples) of each batch           [ 16k ]
        %   n_batch     Number of batches to select             [ 64 ]
        %   overlap     Allowable overlap between batches       [ 0 ]
        %   strict      Force x to meet given dimensions        [ false ]
        %
        % If strict==false and the total data length T <= n_batch*batch_size,
        % then the entire dataset will be returned as a single [T x C] batch,
        % i.e. equivalent to batch_size = T and n_batch = 1.
        [starts, len] = self.planRand(varargin{:});
        x = arrayfun(@(start) {self.read_internal(start,len)}, starts);
        x = cat(3, x{:});
    end
    
    
    function [start, len] = planRand(self, varargin)
        % Randomly select a set of data batches to read
        %   [start, len] = planRand(self, varargin)
        %
        % Returns:
        %   start       [n_batch x 1] index (1..T) of first sample in batch
        %   len         Batch length (same as batch_size)
        % Optional parameters (key/value pairs) [default]:
        %   batch_size  Size (#samples) of each batch           [ 16k ]
        %   n_batch     Number of batches to select             [ 64 ]
        %   overlap     Allowable overlap between batches       [ 0 ]
        %   strict      Force x to meet given dimensions        [ false ]
        %
        % If strict==false and the total data length T <= n_batch*batch_size,
        % then the entire dataset will be returned as a single [T x C] batch,
        % i.e. equivalent to batch_size = T and n_batch = 1.
        ip = inputParser();
        ip.addParameter('batch_size', 16*1024, @isscalar);
        ip.addParameter('n_batch', 64, @isscalar);
        ip.addParameter('overlap', 0, @isscalar);
        ip.addParameter('strict', false, @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        % Collect some dimensions
        len = prm.batch_size; B = prm.n_batch; ovlp = prm.overlap;
        stride = len - ovlp;
        T_all = self.T;
        % Plan out the batches
        total_gap = T_all - B*stride - ovlp;
        if total_gap < 0
            % There isn't enough data for the desired batches
            assert(~prm.strict, self.errid_dim, ['Data is too short to ' ...
                'satisfy the given batchSize, nBatch, overlap specs']);
            offsets = 0;
            len = T_all;
        else
            % Randomly select the batches
            gaps = diff([0; sort(randi([0, total_gap], [B 1]))]);
            offsets = cumsum(stride+gaps) - stride;
        end
        start = offsets + 1;
    end
end

end
