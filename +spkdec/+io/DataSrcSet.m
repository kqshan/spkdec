% DataSrc subclass that consists of a set of underlying DataSrc objects
%
% DataSrcSet properties (read-only):
% Dimensions
%   ext_dim     - Extensible dimension (dimension along which data is read)
%   shape       - Data dimensions, with Inf for the extensible dimension
%   len         - Total length (# of samples) along the extensible dimension
% Underlying data
%   src_set     - {M x 1} underlying DataSrc objects
%   src_len     - [M x 1] length of each underlying source
%   concatable  - Whether this set may be concatenated into a continuous source
%
% DataSrcSet methods:
%   DataSrcSet  - Constructor
%   hasShape    - Return whether this data source has the desired shape
%   read        - Read the requested segment of data
%   readNext    - Read the next segment of data
%   readRand    - Read some randomly-selected batches of data
%   planRand    - Plan a set of reads for random batches of data

classdef DataSrcSet < spkdec.DataSrc

% ---------------------------     Properties     -------------------------------

properties (SetAccess=protected)
    % {M x 1} underlying DataSrc objects that actually contain the data
    src_set
    
    % [M x 1] length of each underlying source
    src_len
    
    % Whether this set may be concatenated into a continuous source
    %
    % If <concatable>==false, then attempting to read over a boundary will
    % produce an error. However, readRand() and planRand() will respect these
    % boundaries and so this can still be used to randomly sample data from this
    % set of sources.
    concatable
end

properties (Constant, Hidden)
    errid_cat = 'spkdec:io:DataSrcSet:NotConcatable';
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = DataSrcSet(src_set, varargin)
        % Construct a new DataSrcSet from a cell array of underlying sources
        %   obj = DataSrcSet(src_set, ...)
        %
        % Required arguments:
        %   src_set    {M x 1} DataSrc objects containing the underlying data
        % Optional parameters (key/value pairs) [default]:
        %   concatable  Whether this set can be concatenated    [ false ]
        src_set = src_set(:);
        assert(iscell(src_set) && ~isempty(src_set) && all(cellfun(@(x) ...
            isa(x,'spkdec.DataSrc'), src_set)), spkdec.DataSrc.errid_arg, ...
            'src_set must be a nonempty cell array of DataSrc objects');
        % Parse optional parameters
        ip = inputParser();
        ip.addParameter('concatable', false, @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        % Make sure they all have the same dimensions
        shapes = cellfun(@(x) {x.shape}, src_set);
        shapes = vertcat(shapes{:});
        assert(all(shapes == shapes(1,:),'all'), spkdec.DataSrc.errid_dim, ...
            'The given DataSrc objects must all have the same shape');
        % Call the superclass constructor
        src_len = cellfun(@(x) x.len, src_set);
        obj = obj@spkdec.DataSrc(shapes(1,:), sum(src_len));
        % Assign child properties
        obj.src_set = src_set;
        obj.src_len = src_len;
        obj.concatable = prm.concatable;
    end
    
    
    % Overload the planRand() method to respect the underlying source boundaries
    
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
        % If strict==false and the total data length <= n_batch*batch_size, then
        % * If self.concatable==true, the entire dataset will be returned as a
        %   single batch.
        % * If self.concatable==false, the number of batches will be reduced as
        %   necessary. An error will be thrown if the number of batches is zero.
        ip = inputParser();
        ip.addParameter('batch_size', 16*1024, @isscalar);
        ip.addParameter('n_batch', 64, @isscalar);
        ip.addParameter('overlap', 0, @isscalar);
        ip.addParameter('strict', false, @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        % Call the superclass if concatable==true
        if self.concatable
            [starts, count] = planRand@spkdec.DataSrc(self, prm);
            return
        end
        % Collect some dimensions
        count = prm.batch_size; B = prm.n_batch; ovlp = prm.overlap;
        stride = count - ovlp;
        src_T = self.src_len;
        src_offset = [0; cumsum(src_T)];
        % See how many batches we can fit into each source
        src_maxB = floor((src_T-count)/stride) + 1;
        valid_sources = find(src_maxB > 0); % [S x 1] valid src idx (1..M)
        S = length(valid_sources);
        assert(~isempty(valid_sources), self.errid_dim, ['Underlying data ' ...
            'sources are too short to allow a single batch of the given size']);
        vs_maxB = src_maxB(valid_sources);  % [S x 1] max #batch per valid src
        % Reduce the number of batches if necessary
        total_maxB = sum(vs_maxB);
        if B > total_maxB
            assert(~prm.strict, self.errid_dim, ['Underlying data sources ' ...
                'are too short for the selected number of batches']);
            B = total_maxB;
        end
        % Randomly determine the number of batches to select from each source
        batch_x = randperm(total_maxB, B)'; % [B x 1] index (1..maxB)
        vs_x = [0; cumsum(vs_maxB)]+1;      % [S x 1] edges (1..maxB+1)
        [~,batch_s] = histc(batch_x, vs_x); % [B x 1] src idx (1..S)
        vs_B = accumarray(batch_s,1,[S 1]); % [S x 1] #batch per valid src
        % We can then ignore the sources with zero batches
        mask = (vs_B > 0);
        valid_sources = valid_sources(mask);
        vs_B = vs_B(mask);
        S = length(valid_sources);
        % Plan out the batches within each source
        starts = cell(S,1);
        for s = 1:S
            src_idx = valid_sources(s);
            rel_starts = self.src_set{src_idx}.planRand('batch_size',count, ...
                'n_batch',vs_B(s), 'overlap',ovlp, 'strict',true);
            starts{s} = src_offset(src_idx) + rel_starts;
        end
        starts = vertcat(starts{:});
    end
end

methods (Access=protected)
    % DataSrcSet implementation of the read_internal() method
    function x = read_internal(self, start, count)
        % Convert this into a set of underlying reads
        last = start + count - 1;
        src_last = cumsum(self.src_len);          % [M x 1] last idx in each src
        src_start = [0; src_last] + 1;            % [M+1 x 1] first idx
        [~,read_src] = histc([start, last], src_start);
        read_src = (read_src(1):read_src(2))';    % [S x 1] srcidx (1..M)
        read_src_len = self.src_len(read_src);
        read_offset = start - src_start(read_src);
        read_offset = max(read_offset, 0);        % [S x 1] offset into each src
        read_last = last - src_start(read_src) + 1;
        read_last = min(read_last, read_src_len); % [S x 1] last idx in each src
        read_start = read_offset + 1;
        read_count = read_last - read_offset;
        % Perform the read
        if length(read_src)==1
            x = self.src_set{read_src}.read(read_start, read_count);
        else
            assert(self.concatable, self.errid_cat, ...
                'This read spans multiple sources but self.concatable==false');
            x_arr = arrayfun(@(i,s,c) {self.src_set{i}.read(s,c)}, ...
                read_src, read_start, read_count);
            x = cat(self.ext_dim, x_arr{:});
        end
    end
end

end
