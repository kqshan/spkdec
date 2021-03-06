% Class for optimizing the spike basis waveforms
%
% BasisOptimizer properties:
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
% BasisOptimizer methods:
%   BasisOptimizer  - Construct a new BasisOptimizer object
%   optimize    - Optimize the spike basis for the given data
% High-level operations
%   makeBasis   - Construct a new SpikeBasis, optimized for the given spikes
%   updateBasis - Update a spike basis using proximal gradient descent
% Object management
%   copy        - Create a deep copy of this handle object
%   saveobj     - Serialize a BasisOptimizer object to struct
%   loadobj     - [Static] Deserialize a BasisOptimizer object from a struct
%
% This class defines the main basis objective function as
%   f(basis) = ||data - basis*spikes||^2,
% i.e. the spike reconstruction error using the whitened inner product. The
% optimize() routine then seeks to solve the following optimization problem:
%     minimize    f(basis) + lambda*||basis-basis_prev||^2
%   subject to    basis is orthonormal

classdef BasisOptimizer < matlab.mixin.Copyable

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
    % The optimization process seeks to minimize the reconstruction error in the
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
    errid_pfx = 'spkdec:BasisOptimizer';
    errid_arg = 'spkdec:BasisOptimizer:BadArg';
    errid_dim = 'spkdec:BasisOptimizer:DimMismatch';
end


% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------


methods
    function obj = BasisOptimizer(whbasis, varargin)
        % BasisOptimizer constructor
        %   obj = BasisOptimizer(whbasis, ...)
        %   obj = BasisOptimizer(whitener, ...)
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
        errid_arg = spkdec.BasisOptimizer.errid_arg;
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
    
    % High-level operations
    [basis, spk] = makeBasis(self, spikes, D, varargin);
    [basis, spk] = updateBasis(self, basis, spk, resid, varargin);
end

% ----------------------     Copy and serialization     ------------------------

methods (Access=protected)
    function obj = copyElement(self)
        obj = copyElement@matlab.mixin.Copyable(self);
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
        obj = spkdec.BasisOptimizer(s.whbasis, rmfield(s,'whbasis'));
    end
end


% ------------------------------------------------------------------------------
% ====================    Protected properties/methods     =====================
% ------------------------------------------------------------------------------


% Individual steps of the optimization -----------------------------------------

methods (Access=protected)
    % Initialization
    A = start_optimization(self, data, prm);
    Y = convert_spikes_to_Y(self, spikes, pad);
    A0 = init_spkbasis(self, D);
    A0 = convert_spkbasis_to_A(self, basis);
    
    % Optimize spikes with basis held constant
    X = optimize_spk(self, A);
    err = eval_error(self, A, X);
    Ar = get_shifted_basis(self, A, r);
    
    % Proximal gradient descent on the basis, with spikes held constant
    grad = compute_gradient(self, A, X);
    A = prox_grad_step(self, A, grad);
    [step_ok, lhs, rhs] = eval_step(self, A, prev_A, X);
    
    % Cleanup
    [basis, spk, resid] = finalize_optimization(self, A, X);
    basis = convert_A_to_spkbasis(self, A);
end

% Higher-level helpers
methods (Access=protected)
    % Reconstruct spike waveforms given detected spikes + residuals
    spikes = reconstruct_spikes(self, basis, spk, resid);
    
    % Append two sets of spike bases (in Q1 coordinates)
    A = append_bases(self, A1, A2);
end


% Temporary cache of problem constants -----------------------------------------

properties (Access=protected, Transient)
    Y           % [L*C x N x S] spike data in Q1 coordinates, with the S
                % dimension corresponding to shifts of (-dt_search:dt_search)
    A0          % [L*C x D] whitened previous basis in Q1 coordinates
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
        norm_XXt = 0;
        for r = 1:self.R
            norm_XXt = norm_XXt + norm(X.X_cov(:,:,r) * X.X_cov(:,:,r)');
        end
        L_max = norm_XXt + self.lambda;
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

% Live updates of the basis waveforms ------------------------------------------

properties (Access=protected, Transient)
    live_lh     % [D x 1] Line handles for the plotted basis waveforms
    live_delay  % Animation delay for the live updates
    live_space  % Inter-channel spacing for the plots
end
methods (Access=protected)
    live_init(self, A, prm);
    live_update(self, A);
    [x,y] = convert_A_to_plot_coords(self, A);
    
    function live_cleanup(self)
        % Cleanup the temporary variables used in the live basis updates
        self.live_lh = []; self.live_delay = []; self.live_space = [];
    end
end

end
