% DataSrc subclass that performs linear interpolation over specified indices
%
% DataSrcStitched properties (read-only)
% Dimensions
%   ext_dim     - Extensible dimension (dimension along which data is read)
%   shape       - Data dimensions, with Inf for the extensible dimension
%   len         - Total length (# of samples) along the extensible dimension
% Underlying data
%   src         - Underlying DataSrc that actually stores the data
%   stitch_idx  - [N x 2] data indices to "stitch" (linearly interpolate) over
%
% DataSrcStitched methods:
%   DataSrcStiched - Constructor
%   hasShape    - Return whether this data source has the desired shape
%   read        - Read the requested segment of data
%   readNext    - Read the next segment of data
%   readRand    - Read some randomly-selected batches of data
%   planRand    - Plan a set of reads for random batches of data


classdef DataSrcStitched < spkdec.DataSrc

% ---------------------------     Properties     -------------------------------

properties (SetAccess=protected)
    % Underlying DataSrc that actually stores the data
    src
    
    % [N x 2] data indices to stitch over
    %
    % These intervals must be sorted and disjoint. For each interval, if we let
    %   first = stitch_idx(i,1)
    %   last  = stitch_idx(i,2)
    % then this replaces the data interval (first:last) with a linear
    % interpolation between (first-1) and (last+1).
    stitch_idx
end

properties (Access=protected)
    % [2*N+2 x 1] vector of stitch interval edges
    %
    % This is constructed as (where si = stitch_idx):
    %   [ -Inf  si(1,1)  si(1,2)+1  si(2,1)  ...  si(N,2)+1  Inf ]'
    % It can be used with histc() to determine which intervals overlap a given
    % read request:
    %   [~,edge_idx] = histc(data_idx, edges);
    %   is_in_interval = (mod(edge_idx,2) == 0);
    %   interval_idx = edge_idx/2;
    edges
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = DataSrcStitched(src, stitch_idx)
        % Construct a new DataSrcStitched from the given stitch intervals
        %   obj = DataSrcStitched(src, stitch_idx)
        %
        % Required arguments:
        %   src         DataSrc object to use as the underlying data store
        %   stitch_idx  [N x 2] data indices to stitch over
        errid_arg = spkdec.DataSrc.errid_arg;
        % Validate the stitch intervals
        x = stitch_idx;
        [N,dim_2] = size(x);
        assert(dim_2==2 && all(mod(x,1)==0,'all'), errid_arg, ...
            'stitch_idx must be an [N x 2] array of data indices');
        assert(all(x > 1,'all') && all(x < src.len,'all'), errid_arg, ...
            'stitch_idx must lie within the range of the underlying source');
        i1 = x(:,1); i2 = x(:,2);
        assert(all(i2 >= i1) && issorted(i1), errid_arg, ['Each row of ' ...
            'stitch_idx must be [first,last] and the rows must be sorted']);
        assert(all(i1(2:N) - i2(1:N-1) > 1), errid_arg, ['The stitch_idx ' ...
            'intervals must be disjoint with at least 1 sample between them']);
        % Construct the object
        obj = obj@spkdec.DataSrc(src.shape, src.len);
        obj.src = src;
        obj.stitch_idx = stitch_idx;
        % Compute the protected property
        edges = [i1'; i2'+1];
        edges = [-Inf; edges(:); Inf];
        obj.edges = edges;
    end
end

methods (Access=protected)
    % DataSrcStitched implementation of the read_internal() method
    function x = read_internal(self, start, count)
        % Get the stitch intervals that overlap this read range
        last = start-1 + count;
        data_range = [start, last];
        [~,edge_range] = histc(data_range, self.edges);
        i1 = ceil(edge_range(1)/2);     % First stitch interval
        i2 = floor(edge_range(2)/2);    % Last stitch interval
        if i2 < i1
            % Special case if there are no stitch intervals for this read
            x = self.src.read(start, count);
            return
        end
        si = self.stitch_idx(i1:i2,:) - (start-1);
        % Read the data, extending the range if necessary
        output_offset = 1 - min(1, si(1,1)-1);
        start_ext = start - output_offset;
        count_ext = max(count, si(end,2)+1) + output_offset;
        x = self.src.read(start_ext, count_ext);
        % Permute so that the extensible dimension is the first dimension
        dims = self.shape;
        ndims = length(dims);
        edim = self.ext_dim;
        perm = [edim, 1:edim-1, edim+1:ndims];
        if ~isequal(perm, 1:ndims)
            x = permute(x, perm);
        end
        % Perform the interpolation
        for k = 1:size(si,1)
            t1 = si(k,1)+output_offset; t2 = si(k,2)+output_offset;
            alpha = (1:t2-t1+1)' / (t2-t1+2);
            x(t1:t2,:) = alpha*x(t2+1,:) + (1-alpha)*x(t1-1,:);
        end
        % Extract the relevant range if necessary
        if (count_ext ~= count)
            x = x(output_offset+(1:count),:);
            if ndims > 2
                perm_dims = dims(perm);
                perm_dims(1) = count;
                x = reshape(x, perm_dims);
            end
        end
        % Permute back
        perm2 = [(1:edim-1)+1, 1, edim+1:ndims];
        if ~isequal(perm2, 1:ndims)
            x = permute(x, perm2);
        end
    end
end

methods (Static, Hidden)
    function test()
        % Run unit tests on this class
        %   spkdec.io.DataSrcStitched.test()
        T = 1000;
        case_dims = {Inf, [Inf 4 3], [4 Inf 3], [4 3 Inf]};
        for case_idx = 1:length(case_dims)
            % Construct some test data
            orig_src = test_create_data(case_dims{case_idx}, T);
            [nan_src, stitch_idx] = test_create_stitch_intervals(orig_src);
            % Create the stitched source
            stitch_src = spkdec.io.DataSrcStitched(nan_src, stitch_idx);
            % Compare this to the original data. It should match since the
            % original data is linear along the extensible dimension.
            batch_sz = 50;
            is_last = false;
            while ~is_last
                [x, is_last] = stitch_src.readNext(batch_sz);
                x_ref = orig_src.readNext(batch_sz);
                assert(all(abs(x-x_ref) < 1e-12,'all'), 'Unit test failed');
            end
        end
        disp('spkdec.io.DataSrcStitched unit tests successful');
    end
end

end



% ------------------------     Helper functions     ----------------------------

function src = test_create_data(shape, len)
% Create data for our unit tests
%   src = test_create_data(shape, len)
%
% Returns:
%   src         DataSrcMatrix that is linear along the extensible dimension
% Required arguments:
%   shape       Desired data source shape
%   len         Length along the extensible dimension
edim = find(isinf(shape));
% Create the data array
sequence = shiftdim(1:len, 2-edim);
dims = shape;
dims(edim) = 1;
offsets = rand([dims, 1]);
data = offsets + sequence;
% Wrap this in a DataSrcMatrix
src = spkdec.io.DataSrcMatrix(data, edim, length(shape));
end


function [nan_src, stitch_idx] = test_create_stitch_intervals(orig_src)
% Create a modified source with NaNs and stitch indices to test
%   [nan_src, stitch_idx] = test_create_stitch_intervals(orig_src)
%
% Returns:
%   nan_src     DataSrcMatrix with NaN values in stitch intervals
%   stitch_idx  [N x 2] indices to stitch over
% Required arguments:
%   orig_src    DataSrcMatrix containing the original data
X = orig_src.data;
ndims = length(orig_src.shape);
edim = orig_src.ext_dim;
len = orig_src.len;
% Decide on the stitch intervals
n_intervals = 40;
idx = sort(randperm(len-1, 2*n_intervals))+1;
idx = reshape(idx,[2, n_intervals]);
% Make sure we test some edge cases
idx(2,1) = idx(1,1)+1;
idx(2,5) = idx(1,6)-1;
% Determine which data samples are within a stitch interval
[~,bin] = histc((1:len)', [-Inf; idx(:); Inf]);
is_in_int = (mod(bin,2) == 0);
% Set those values to NaN
S = struct('type','()');
S.subs = repmat({':'}, [1,ndims]);
S.subs{edim} = is_in_int;
X = subsasgn(X, S, NaN); % X(:,is_in_int) = NaN
% Create the new DataSrc
nan_src = spkdec.io.DataSrcMatrix(X, edim, ndims);
% And the stitch_idx for DataSrcStitched
stitch_idx = [idx(1,:)', idx(2,:)'-1];
end
