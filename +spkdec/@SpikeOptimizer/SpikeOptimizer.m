% Class for optimizing the spike basis waveforms
%
% SpikeOptimizer properties:
% Dimensions
%   C           - Number of data channels
%   W           - Whitening filter length (#samples)
%   L           - Waveform length (#samples) before whitening
%   R           - Interpolation ratio (1 = no interpolation)
% Data
%   whbasis     - WhitenerBasis object that defines the space of waveforms
% Optimization parameters (publicly-settable)
%   dt_search   - Spike time jitter (#samples) to search over during optim.
%   n_iter      - Number of alternating descent steps to take
%   verbose     - Print status updates to stdout
%
% SpikeOptimizer methods:
%   SpikeOptimizer  - Construct a new SpikeOptimizer object
%   optimize    - Find a basis that minimizes reconstruction error
%   optimizeCS  - Find a channel-specific basis that minimizes rec. error
% High-level operations
%   makeBasis   - Construct a new SpikeBasis, optimized for the given spikes
%   updateBasis - Update a spike basis using proximal gradient descent
% Object management
%   copy        - Create a deep copy of this handle object
%   saveobj     - Serialize a SpikeOptimizer object to struct
%   loadobj     - [Static] Deserialize a SpikeOptimizer object from a struct

classdef SpikeOptimizer < matlab.mixin.Copyable

% ------------------------------------------------------------------------------
% ========================     Public properties     ===========================
% ------------------------------------------------------------------------------


properties (SetAccess=private, Dependent)
    % Number of data channels
    C
    
    % Whitening filter length (#samples)
    % The overall length of the whitener waveform is therefore L+W-1
    W
    
    % Waveform length (#samples) before whitening
    %
    % This optimization relies on explicit bases for the whitening operation
    % (see also spkdec.WhitenerBasis), which requires a pre-determined waveform
    % support.
    L
    
    % Sub-sample interpolation ratio (1 = no interpolation)
    R
end
methods
    function val = get.C(self), val = self.whbasis.C; end
    function val = get.W(self), val = self.whbasis.W; end
    function val = get.L(self), val = self.whbasis.L; end
    function val = get.R(self), val = self.whbasis.R; end
end

properties (SetAccess=protected)
    % WhitenerBasis object that defines the space of waveforms
    %
    % The optimization process seeks to minimize the reconstriction error in the
    % whitened space defined by this WhitenerBasis object. This specifies both
    % the whitening (whbasis.whitener) and the sub-sample interpolation
    % (whbasis.interp) that will be applied to the spike basis waveforms.
    whbasis
end

properties
    % Spike timing jitter (#samples) to search over during optimization
    %
    % During the optimization process, we are jointly optimizing the basis
    % waveforms and the detected spike properties. This includes not only the
    % spikes' feature space representation, but also their sub-sample shift
    % index and (if dt_search > 0) minor error in the detected spike times. 
    % This latter optimization involves an exhaustive search, so don't make
    % dt_search too large.
    dt_search
    
    % Number of alternating descent steps to take
    %
    % Rather than check for convergence, this just runs a fixed number of
    % iterations. Convergence is usually irrelevant since this optimization is
    % usually performed as part of a stochastic optimization.
    n_iter
    
    % Print status updates to stdout
    %
    % You can choose from a range of verbosity levels:
    %   0,false     No output
    %   1,true      Summary only
    %   1<v<=2      Update every 1/(v-1) iterations
    verbose
end

properties (Constant, Hidden)
    errid_arg = 'spkdec:SpikeOptiimzer:BadArg';
    errid_dim = 'spkdec:SpikeOptimizer:DimMismatch';
end


% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------


methods
    function obj = SpikeOptimizer(whbasis, varargin)
        % SpikeOptimizer constructor
        %   obj = SpikeOptimizer(whbasis, ...)
        %   obj = SpikeOptimizer(whitener, ...)
        %
        % Required arguments
        %   whbasis   WhitenerBasis object (specifies both whitener and interp)
        % --- or ---
        %   whitener  Whitener object (interp must be provided as param below)
        %
        % Optional parameters (key/value pairs) [default]:
        %   interp    Interpolator object                       [ none ]
        %   dt_search Spike timing error to search over         [ 1 ]
        %   n_iter    Number of iterations in optimization      [ 100 ]
        %   verbose   Print status updates to stdout            [ false ]
        errid_arg = spkdec.SpikeOptimizer.errid_arg;
        % Parse the additional arguments
        ip = inputParser();
        ip.addParameter('interp', [], ...
            @(x) isempty(x) || isa(x,'spkdec.Interpolator'));
        ip.addParameter('dt_search', 1, @isscalar);
        ip.addParameter('n_iter', 100, @isscalar);
        ip.addParameter('verbose', false, @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        % Construct the WhitenerBasis
        if isa(whbasis,'spkdec.Whitener')
            interp = prm.interp;
            assert(~isempty(interp), errid_arg, ['If the first argument ' ...
                'is a Whitener object, then <interp> must be provided']);
            whbasis = spkdec.WhitenerBasis(whbasis, 'interp',interp);
        end
        assert(isa(whbasis,'spkdec.WhitenerBasis'), errid_arg, ...
            'The first argument must be a WhitenerBasis or a Whitener object');
        % Assign properties
        obj.whbasis = whbasis;
        for fn = {'dt_search','n_iter','verbose'}
            obj.(fn{1}) = prm.(fn{1});
        end
    end
    
    
    % Main optimization routine
    [basis, spk, resid] = optimize(self, spikes, varargin);
    [basis_cs, spk, resid] = optimizeCS(self, spikes, varargin);
    
    % High-level operations
    [basis, spk] = makeBasis(self, spikes, D, varargin);
    [basis, spk] = updateBasis(self, basis, spk, resid, varargin);
end

% ----------------------     Copy and serialization     ------------------------

methods (Access=protected)
    function obj = copyElement(self)
        obj = copyElement@matlab.mixin.copyable(self);
        obj.whbasis = copy(self.whbasis);
    end
end
methods
    function s = saveobj(self)
        s = struct();
        for fn = {'whbasis','dt_search','n_iter','verbose'}
            s.(fn{1}) = self.(fn{1});
        end
        s.whbasis = s.whbasis.saveobj();
    end
end
methods (Static)
    function obj = loadobj(s)
        obj = spkdec.SpikeOptimizer(s.whbasis, rmfield(s,'whbasis'));
    end
end


% ------------------------------------------------------------------------------
% ====================    Protected properties/methods     =====================
% ------------------------------------------------------------------------------


% Individual steps of the optimization -----------------------------------------

methods (Access=protected)
    % Initialization
    Y = convert_spikes_to_Y(self, spikes, pad);
    A0 = init_spkbasis(self, D);
    
    % Optimize spikes with basis held constant
    X = optimize_spk(self, A);
    err = eval_error(self, A, X);
    Ar = get_shifted_basis(self, A, r);
    
    % Proximal gradient descent on the basis, with spikes held constant
    grad = compute_gradient(self, A, X);
    A = prox_grad_step(self, A, grad);
    step_ok = eval_step(self, A, prev_A, X);
    
    % Convert from the whitened Q2 coordinates back to raw waveforms
    basis = convert_A_to_spkbasis(self, A);
end

% Other helpers
methods (Access=protected)
    spikes = reconstruct_spikes(self, basis, spk, resid);
end


% Temporary cache of problem constants -----------------------------------------

properties (Access=protected, Transient)
    Y           % [L*C x N x S] spike data in the Q1 basis, with the S dimension
                % corresponding to shifts of (-dt_search:dt_search)
    basis_mode  % Basis constraints: {'channel-specific','omni-channel'}
    A0          % Whitened previous basis. Dimensions depend on basis_mode:
                %   channel-specific  [L x K x C] in Q2 basis
                %   omni-channel      [L*C x D] in Q1 basis
    lambda      % Weight applied to the proximal regularizer ||A-A0||
    spk_r       % User-specified sub-sample shift for each spike
    t_start     % tic() when we started this problem
end


% Backtracking -----------------------------------------------------------------

properties (Access=protected, Transient)
    lip         % Current value of the local Lipcshitz estimate
    lip_max     % Estimated maximum value of <lip> based on linear operator
    lip_min     % Minimum value of <lip> based on lambda
    nBack       % Number of backtracking steps on this iteration
end
properties (Access=protected, Constant)
    lip_up = 2;       % Factor to increase <lip> by when backtracking
    lip_decay = 1.2;  % Factor to decay <lip> by on each iteration
end
methods (Access=protected)
    function lipschitz_init(self, X)
        % Initialize the Lipschitz estimate (requires the spikes X)
        %   lipschitz_init(self, X)
        switch self.basis_mode
            case 'channel-specific'
                norm_map21 = norm(self.whbasis.map_21(:,:));
            case 'omni-channel'
                norm_map21 = 1;
        end
        norm_XXt = 0;
        for r = 1:self.R
            norm_XXt = norm_XXt + norm(X.X_cov(:,:,r) * X.X_cov(:,:,r)');
        end
        L_max = norm_map21^2*norm_XXt + self.lambda;
        self.lip_max = L_max; self.lip_min = self.lambda;
        self.lip = (self.lip_max+self.lip_min)/2; self.nBack = 0;
    end
    
    function lipschitz_decay(self)
        % Decay the Lipschitz estimate, increasing the step size
        self.lip = (self.lip-self.lip_min) / self.lip_decay + self.lip_min;
        self.nBack = 0;
    end
    
    function lipschitz_backtrack(self)
        % Update the Lipschitz estimate during backtracking (reduce step size)
        self.lip = (self.lip-self.lip_min) * self.lip_up + self.lip_min;
        self.nBack = self.nBack + 1;
    end
    
    function lipschitz_cleanup(self)
        % Cleanup the temporary variables used in backtracking
        self.lip = []; self.lip_max = []; self.lip_min = []; self.nBack = [];
    end
end


% Verbose output ---------------------------------------------------------------

properties (Access=protected, Transient)
    vb_format   % Format string for iteration-level output
    vb_period   % Number of iterations between updates
    vb_last     % Iteration # of the last update
    norm_y      % Squared Frobenius norm of Y
    err_0       % Reconstruction error on the 0th iteration
end
methods (Access=protected)
    verbose_init(self);
    verbose_update(self, iter, A, X);
    
    function verbose_cleanup(self)
        % Cleanup the temporary variables used in the verbose output
        self.vb_format = []; self.vb_period = []; self.vb_last = [];
        self.norm_y = []; self.err_0 = [];
    end
end

end
