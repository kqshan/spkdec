% Gram matrices (dot products) for a set of kernels at different lags
%
% Gramians properties:
% Dimensions
%   L           - Kernel length (#samples)
%   C           - Number of data channels
%   D           - Number of kernels per family
%   R           - Number of kernel families
% Data
%   kernels     - [L x C x D x R] kernels under consideration
%
% Gramians methods:
%   Gramians    - Construct a new Gramians object
%   getGram     - Return the Gram matrix at a specified lag
%   getGramSeq  - Return the Gram matrix for a sequence of lags
%   lagNorms    - Return a vector of norm(self.getGram(lag)) at lag = 0..L-1
% Object management
%   copy        - Create a copy of this handle object
%   saveobj     - Serialize a Gramians object to struct
%   loadobj     - [Static] Deserialize a Gramians object from a struct

classdef Gramians < matlab.mixin.Copyable

% ------------------------------------------------------------------------------
% ========================     Public properties     ===========================
% ------------------------------------------------------------------------------


properties (SetAccess=private, Dependent)
    % Kernel length (#samples)
    L
    
    % Number of data channels
    C
    
    % Number of kernels per family
    % If comparing this to classes like SpikeBasis or Convolver, then D = K*C
    D
    
    % Number of kernel families
    % One way that kernel families arise is through the sub-sample interpolation
    % option in spkdec.SpikeBasis, so that's why this is named R.
    R
end
methods
    function val = get.L(self), val = size(self.kernels,1); end
    function val = get.C(self), val = size(self.kernels,2); end
    function val = get.D(self), val = size(self.kernels,3); end
    function val = get.R(self), val = size(self.kernels,4); end
end

properties (SetAccess=protected)
    % [L x C x D x R] kernels under consideration
    %
    % This contains D*R kernels grouped into R families of D kernels each. Each
    % kernel is an [L x C] matrix (the dot products among them are the Frobenius
    % inner product) and the "lag" option applies to the L dimension.
    kernels
end

properties (Constant, Access=private)
    errid_arg = 'spkdec:Gramians:BadArg';
    errid_dim = 'spkdec:Gramians:DimMismatch';
end


% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------


methods
    function obj = Gramians(kernels, varargin)
        % Gramians constructor
        %   obj = Gramians(kernels, ...)
        %
        % Required arguments:
        %   kernels   [L x C x D x R] kernels under consideration
        obj.kernels = gather(double(kernels));
        % Initialize the cache
        [L,~,D,R] = size(kernels);
        max_cache_mb = D; % Just a heuristic; maybe should be user-configurable
        cache_size = round(max_cache_mb*pow2(20) / (8*D*D));
        cache_size = min(cache_size, R*R*L);
        obj.cache_init(cache_size);
    end
    
    % Return Gram matrices
    mat = getGram(self, lag, r1, r2, skip_cache);
    bands = getGramSeq(self, spk_t, spk_r, varargin);
    
    % Compute norms of Gram matrices
    norms = lagNorms(self);
end

methods (Static)
    % Unit tests
    test(varargin);
end


% ----------------------     Copy and serialization     ------------------------

methods
    function s = saveobj(self)
        s = struct();
        s.kernels = self.kernels;
    end
end
methods (Static)
    function obj = loadobj(s)
        obj = spkdec.Gramians(s.kernels);
    end
end


% ------------------------------------------------------------------------------
% ===========================      Protected     ===============================
% ------------------------------------------------------------------------------


% Output of lagNorms()
properties (Access=protected)
    lag_norms   % [L x 1] norm(self.getGram(lag)) for lag = 0..L-1
end


% Helper functions for getGramSeq()
methods (Access=protected)
    G_blkband = get_gram_for_overlaps(self, overlaps);
end
methods (Static, Access=protected)
    overlaps = find_overlaps(t, r, dt_max);
    G_bands = blkband_to_bands(G_blkband);
end


% Gram matrix cache (gmc)
properties (Access=protected)
    gmc_lookup  % [R x R x L] cached matrix index (1..N) or 0 if not cached
    gmc_revidx  % [N x 1] lookup index (1..R*R*L) of cache contents
    gmc_data    % [D x D x N] cache of Gram matrices
    gmc_size    % Cache size (N)
    gmc_count   % Number of matrices currently in the cache
end
methods (Access=protected)
    function cache_init(self, N_max)
        % Initialize the Gram matrix cache
        %   cache_init(self, N_max)
        %
        % Required arguments:
        %   N_max   Cache size (# of matrices)
        self.gmc_lookup = zeros(self.R, self.R, self.L, 'uint32');
        self.gmc_revidx = zeros(N_max, 1, 'uint32');
        self.gmc_data = zeros(self.D, self.D, N_max);
        self.gmc_size = N_max;
        self.gmc_count = 0;
    end
    
    function mat = cache_request(self, r1, r2, lag)
        % Request a Gram matrix from the cache
        %   mat = cache_request(r1, r2, lag)
        %
        % Returns:
        %   mat     [D x D] cached Gram matrix, or [] if not cached
        % Required arguments:
        %   r1,r2   Kernel family indices (1..R)
        %   lag     Lag between kernels (0..L-1)
        idx = self.gmc_lookup(r1, r2, lag+1);
        if idx==0
            mat = [];
        else
            mat = self.gmc_data(:,:,idx);
        end
    end
    
    function cache_write(self, r1, r2, lag, mat)
        % Write the given Gram matrix to the cache
        %   cache_write(r1, r2, lag, mat)
        %
        % Required arguments:
        %   r1,r2   Kernel family indices (1..R)
        %   lag     Lag between kernels (0..L-1)
        %   mat     [D x D] Gram matrix to cache
        R_ = self.R;
        rrl = r1 + R_*(r2-1) + R_*R_*(lag); % Remember, it's (r1, r2, lag+1)
        idx = self.gmc_lookup(rrl);
        if idx==0
            % Decide where to place this new cache entry
            if self.gmc_count < self.gmc_size
                % Just append it
                idx = self.gmc_count+1;
                self.gmc_count = idx;
            else
                % Evict a random entry from the cache
                % This eviction policy could probably be improved
                idx = randi(self.gmc_size);
                rrl_evict = self.gmc_revidx(idx);
                self.gmc_lookup(rrl_evict) = 0;
            end
            % Update the cache
            self.gmc_data(:,:,idx) = mat;
            self.gmc_lookup(rrl) = idx;
            self.gmc_revidx(idx) = rrl;
        else
            % Replace existing cache entry
            self.gmc_data(:,:,idx) = mat;
        end
    end
end

end
