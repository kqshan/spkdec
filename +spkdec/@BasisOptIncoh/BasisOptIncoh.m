% BasisOptimizer subclass that applies a penalty on self-coherence
%
% BasisOptIncoh properties:
% Dimensions
%   C           - Number of data channels
%   W           - Whitening filter length (#samples)
%   L           - Waveform length (#samples) before whitening
%   R           - Interpolation ratio (1 = no interpolation)
% Data
%   whbasis     - WhitenerBasis object that defines the space of waveforms
% Optimization parameters (publicly-settable)
%   coh_penalty - Coherence penalty weight
%   scaling_ok  - Ignore isotropic scaling in the coherence penalty
%   dt_search   - Spike time jitter (#samples) to search over during optim.
%   n_iter      - Number of alternating descent steps to take
%   verbose     - Print status updates to stdout
%
% BasisOptIncoh methods:
%   BasisOptIncoh  - Construct a new BasisOptIncoh object
%   optimize    - Optimize the spike basis for the given data
% High-level operations
%   makeBasis   - Construct a new SpikeBasis, optimized for the given spikes
%   updateBasis - Update a spike basis using proximal gradient descent
% Object management
%   copy        - Create a deep copy of this handle object
%   saveobj     - Serialize a BasisOptIncoh object to struct
%   loadobj     - [Static] Deserialize a BasisOptIncoh object from a struct
%
% This class defines the objective function as
%   f(basis) = ||data - basis*spikes||^2 + coh_penalty*g(basis).
% The first term is the squared reconstruction error, which is the same as in
% the parent class (BasisOptimizer), and the second term is a regularizer that
% penalizes off-center "activation" of the basis with respect to the data:
%   g(basis) = 1/T * sum_t sum_n <basis, shift(t)*data(:,n)>^2
% This is related to coherence since we are simultaneously trying to minimize
% the reconstruction error, which requires maximizing <basis,data> without
% the shift term. The addition of this coherence penalty therefore incentivizes
% spike bases that can represent the given data but don't "respond" to shifted
% versions of the data.
%
% The optimize() routine then seeks to solve the following optimization problem:
%     minimize    f(basis) + lambda*||basis-basis_prev||^2
%   subject to    basis is orthonormal


classdef BasisOptIncoh < spkdec.BasisOptimizer

% ------------------------------------------------------------------------------
% ========================     Public properties     ===========================
% ------------------------------------------------------------------------------


properties
    % Coherence penalty weight
    %
    % The objective function being optimized by this class is:
    %   f(basis) = ||data - basis*spikes||^2 + coh_penalty*g(basis)
    % where g(basis) is a regularizer that penalizes the self-coherence of the
    % fitted basis with respect to the detected spikes. Increasing the value
    % of `coh_penalty` will place greater priority on the goal of having an
    % incoherent basis.
    coh_penalty
end


% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------


methods
    function obj = BasisOptIncoh(whbasis, varargin)
        % BasisOptIncoh constructor
        %   obj = BasisOptIncoh(whbasis, ...)
        %   obj = BasisOptIncoh(whitener, ...)
        %
        % Required arguments
        %   whbasis     WhitenerBasis object (specifies whitener and interp)
        % --- or ---
        %   whitener    Whitener object (interp must be provided as param below)
        %
        % Optional parameters (key/value pairs) [default]:
        %   coh_penalty Coherence penalty weight                [ 0.1 ]
        %   ...         Add'l param are forwarded to superclass constructor
        
        % Parse the additional arguments
        ip = inputParser();
        ip.KeepUnmatched = true; ip.PartialMatching = false;
        ip.addParameter('coh_penalty', 0.1, @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        addl_prm = ip.Unmatched;
        % Call the superclass constructor
        obj = obj@spkdec.BasisOptimizer(whbasis, addl_prm);
        % Assign the new properties
        for fn = fieldnames(prm)'
            obj.(fn{1}) = prm.(fn{1});
        end
    end
    
    % Main optimization routine
    [basis, spk, resid] = optimize(self, spikes, varargin);
end

% ----------------------     Copy and serialization     ------------------------

methods
    function s = saveobj(self)
        s = saveobj@spkdec.BasisOptimizer(self);
        for fn = {'coh_penalty'}
            s.(fn{1}) = self.(fn{1});
        end
    end
end
methods (Static)
    function obj = loadobj(s)
        obj = spkdec.BasisOptIncoh(s.whbasis, rmfield(s,'whbasis'));
    end
end


% ------------------------------------------------------------------------------
% ====================    Protected properties/methods     =====================
% ------------------------------------------------------------------------------


% Individual steps of the optimization -----------------------------------------

methods (Access=protected)
    % Modified to also compute/cleanup coh_YYt, coh_L
    A = start_optimization(self, data, prm);
    [basis, spk, resid] = finalize_optimization(self, A, X);
    
    % Modified to include the coherence regularizer in the f() term
    grad = compute_gradient(self, A, X);
    [step_ok, lhs, rhs] = eval_step(self, A, prev_A, X);
end

% Temporary cache of problem constants -----------------------------------------

properties (Access=protected, Transient)
    coh_YYt     % [L*C x L*C] mean shifted spike covariance (Y*Y') in Q1 coords
                %   coh_YYt = sum_t(w_t*shift_t*Y*Y'*shift_t)/sum_t(w_t)
    coh_L       % Cholesky decomposition of coh_YYt: coh_L*coh_L' == coh_YYt
end

% Backtracking -----------------------------------------------------------------

methods (Access=protected)
    function lipschitz_init(self, X)
        lipschitz_init@spkdec.BasisOptimizer(self, X);
        % Add coh_YYt to lip_min/lip_max
        s = svd(self.coh_YYt);
        self.lip_min = self.lip_min + min(s);
        self.lip_max = self.lip_max + max(s);
        % Re-evaluate the starting Lipschitz estimate
        self.lip = (self.lip_max+self.lip_min)/2;
    end
end

% Verbose output ---------------------------------------------------------------

methods (Access=protected)
    % Modified to include the coherence regularizer in the output
    verbose_init(self);
    verbose_update(self, iter, A, X);
end

end
