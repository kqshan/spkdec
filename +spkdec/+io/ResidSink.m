% Special DataSink for spike-centered residuals
%
% ResidSink properties:
% Dimensions (read-only)
%   ext_dim     - Extensible dimension (fixed as 3)
%   shape       - Data sink dimensions (fixed as [L+W-1, C, Inf])
%   len         - Current length along the extensible dimension
% Data
%   whbasis     - WhitenerBasis used to unwhiten the given residuals
%   out_thresh  - Threshold at which to classify a residual as an outlier
% Underlying DataSinks (may be empty)
%   given       - [L+W-1 x C x N] whitened residuals
%   norm        - [N] residual norm in attainable subspace
%   unwhitened  - [L x C x N] unwhitened residuals
%   outlier_idx - [m] indices (1..N) of identified outliers
%   outlier     - [L+W-1 x C x m] whitened residuals of identified outliers
%
% ResidSink methods:
%   ResidSink   - Constructor
%   hasShape    - Return whether this data sink has the desired shape
%   append      - Write additional residuals to the end of this data sink
%   asMatrices  - [Static] Make a new ResidSink with DataSinkMatrix datasinks
%
% The ResidSink object appears as a [L+W-1 x C x Inf] DataSink and is compatible
% with the 'resid' output of util.detect_spikes(). It receives whitened spike-
% centered residuals and computes a number of derived quantities, which are then
% written to its underlying DataSinks.
%
% See also: spkdec.util.detect_spikes, spkdec.WhitenerBasis.unwhiten

classdef ResidSink < spkdec.DataSink

% ---------------------------     Properties     -------------------------------

properties
    % WhitenerBasis used to unwhiten the given residuals
    %
    % This contains a decomposition of the whitening operation that allows us to
    % more efficiently project a given unwhitened residual onto the span of the
    % whitener (for the <norm> output) and evaluate the whitener pseudoinverse
    % (for the <unwhitened> output).
    whbasis
    
    % Threshold at which to classify a residual as an outlier
    %
    % For each residual, we compare its norm (as computed for the <norm> output)
    % to this threshold to determine whether it should be classified as an
    % outlier.
    out_thresh
end

properties
    % [L+W-1 x C x N] whitened residuals
    %
    % This simply stores the given residuals. Probably not useful unless you
    % want to chain up a bunch of DataSinks into a directed acyclic graph.
    given = []
    
    % [N] residual norm in attainable subspace
    %
    % This stores the norm of the residual, i.e. how much of the observed data
    % is left unaccounted for by our K*C-dimensional sparse-deconvolution-based
    % approximation of the source data.
    %
    % When combined with the spike norm (see SpikeBasis.spkNorms), this can be
    % used to compute metrics like the fraction of variance unexplained. This
    % norm is also used to classify residuals as outliers.
    %
    % What is the "attainable subspace"? The whitened residuals may occuy an
    % [L+W-1 x C] space, but there is only an [L x C] subspace that is actually
    % attainable by whitening a signal of length L. The non-attainable subspace
    % essentially consists of content that falls outside the L window but within
    % the larger L+W-1 window, and it would be unfair to consider these part of
    % the residual norm. This norm is therefore measured after projecting the
    % given residuals onto the [L x C] attainable subspace.
    %
    % See also: spkdec.SpikeBasis.spkNorms, spkdec.WhitenerBasis.unwhiten,
    % spkdec.io.ResidSink.unwhitened, spkdec.io.ResidSink.outlier
    norm = []
    
    % [L x C x N] unwhitened residuals
    %
    % This stores the pseudoinverse of the whitening operator, applied to the
    % given residuals (see WhitenerBasis.unwhiten). In other words, this stores
    % x = argmin ||given_residual - whiten(x)||.
    %
    % This is useful because L+W-1 is often >> L, so this is a more efficient
    % representation of the spike residual. As discussed in ResidSink.norm, the
    % given residual contains components that don't really belong to this spike.
    % 
    % Note that this is different from the "non-whitened" residual, which you
    % might compute by extracting an L-length window around each detected spike
    % in the <data_resid> output of util.detect_spikes. These are both [L x C]
    % spike-centered residuals, but the "non-whitened" residual minimizes the
    % distance in raw data space, whereas this "unwhitened" residual minimizes
    % the distance in whitened space.
    %
    % Also note that <norm>(n) = ||whiten(<unwhitened>(:,:,n)||
    % See also: spkdec.WhitenerBasis.unwhiten, spkdec.io.ResidSink.norm
    unwhitened = []
    
    % [m] indices (1..N) of identified outliers
    %
    % This stores the source indices (1..N, corresponding to the extensible
    % dimension of the <given>, <norm>, and <unwhitened> data sinks) of the
    % residuals that have been classified as outliers.
    %
    % See also: spkdec.io.ResidSink.outlier
    outlier_idx = []
    
    % [L+W-1 x C x m] whitened residuals of identified outliers
    %
    % This stores a copy of the given residuals for the subset of residuals
    % whose norm is large enough that we have classified it as an outlier. The
    % intended use case is to provide more context around detected spikes that
    % are not well-approximated by the feature space representation (e.g.
    % because they are due to overlapping spikes).
    %
    % See also: spkdec.io.ResidSink.outlier_idx, spkdec.io.ResidSink.out_thresh,
    % spkdec.io.ResidSink.norm, spkdec.io.ResidSink.unwhitened
    outlier = []
end

% -----------------------------     Methods     --------------------------------

methods
    function obj = ResidSink(whbasis, varargin)
        % Construct a new ResidSink given a WhitenerBasis and underlying sinks
        %   obj = ResidSink(whbasis, ...)
        %
        % Required arguments:
        %   whbasis     WhitenerBasis used to unwhiten the given residuals
        % Optional parameters (key/value pairs) [default]:
        %   out_thresh  Outlier threshold                   [ sqrt(4*L*C) ]
        %   ...         Additional key/value pairs should provide underlying
        %               DataSinks using the property name as the key string
        L = whbasis.L; W = whbasis.W; C = whbasis.C;
        % Parse key/value arguments
        ip = inputParser();
        ip.KeepUnmatched = true; ip.PartialMatching = false;
        ip.addParameter('out_thresh', sqrt(4*L*C), @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        addl_args = ip.Unmatched;
        % Assign values to self
        obj = obj@spkdec.DataSink([L+W-1, C, Inf]);
        obj.whbasis = whbasis;
        obj.out_thresh = prm.out_thresh;
        for fn = fieldnames(addl_args)'
            obj.(fn{1}) = addl_args.(fn{1});
        end
    end
end

methods (Static)
    function obj = asMatrices(whbasis)
        % Make a new ResidSink using DataSinkMatrix objects as underlying sinks
        %   obj = ResidSink.asMatrices(whbasis)
        %
        % Returns:
        %   obj         New ResidSink object
        % Required arguments:
        %   whbasis     WhitenerBasis used to unwhiten the given residuals
        L = whbasis.L; W = whbasis.W; C = whbasis.C;
        make_sink = @(shape) spkdec.io.DataSinkMatrix(shape);
        obj = spkdec.io.ResidSink(whbasis, 'given',make_sink([L+W-1,C,Inf]), ...
            'norm',make_sink(Inf), 'unwhitened',make_sink([L,C,Inf]), ...
            'outlier_idx',make_sink(Inf), 'outlier',make_sink([L+W-1,C,Inf]) );
    end
end

methods (Access=protected)
    % ResidSink implementation of the append_internal() method
    function append_internal(self, data)
        % We assume that data is a [L+W-1 x C x N_new] 
        N_new = size(data,3);
        idx_offset = self.len;
        % Start with the easy ones
        [unwh, resid_norm] = self.whbasis.unwhiten(data);
        if ~isempty(self.given),        self.given.append(data);        end
        if ~isempty(self.norm),         self.norm.append(resid_norm);   end
        if ~isempty(self.unwhitened),   self.unwhitened.append(unwh);   end
        % Outliers
        is_outlier = (resid_norm > self.out_thresh);
        if any(is_outlier)
            out_idx = find(is_outlier);
            if ~isempty(self.outlier_idx)
                self.outlier_idx.append(out_idx + idx_offset);
            end
            if ~isempty(self.outlier)
                self.outlier.append(data(:,:,out_idx));
            end
        end
        % Increment the length
        self.len = self.len + N_new;
    end
end

end
