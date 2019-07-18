% Greedy solver (orthogonal matching pursuit) for sparse deconvolution problems
%
% Solver properties:
%   det_thresh  - Spike detection threshold (relative to D)
%   det_refrac  - Refractory period (#samples) for spike detection
%   coh_thresh  - Mutual coherence threshold used within an OMP iteration
%   gram_thresh - Threshold to consider a Gram matrix negligible
%   verbose     - Print status updates to stdout
%
% Solver methods:
%   Solver      - Constructor
%   solve       - Find x minimizing phi(x) = ||A*x-b||^2 + beta*nnz(x)
%   detect      - Wrapper for solve() that performs additional postprocessing
% Object management
%   copy        - Create a copy of this handle object
%   saveobj     - Serialize a Solver object to struct
%   loadobj     - [Static] Deserialize a Solver object from a struct

classdef Solver < matlab.mixin.Copyable

% ------------------------------------------------------------------------------
% ========================     Public properties     ===========================
% ------------------------------------------------------------------------------


properties
    % Spike detection threshold (relative to the number of basis dimensions D)
    %
    % The solve() method seeks to minimize ||A*x-b||^2 + beta*nnz_cols(x), where
    % A is the convolutional spike basis, x is the column-sparse matrix of spike
    % features, nnz_cols(x) returns the number of nonzero columns in x, and
    % beta is a parameter that controls the sparsity of the solution.
    %
    % Specifically, this will detect a spike only if adding the spike will
    % improve our squared reconstruction error by more than <beta>. If
    % det_refrac == 0, then this also guarantees that ||A'*residual|| < beta.
    %
    % beta = D * det_thresh. This alternative parameterization is useful
    % because ||A'*z||, where z is taken from a standard multivariate normal
    % distribution, is chi-squared distributed with D degrees of freedom, so
    % this can be a more convenient way to think about false positive rates.
    % Default = 10
    det_thresh = 10;
    
    % Refractory period (#samples) for spike detection
    %
    % This solver will not detect any spikes within +/- det_refrac samples of
    % another spike. Default = 0
    det_refrac = 0;
    
    % Mutual coherence threshold used within an OMP iteration
    %
    % We will detect multiple spikes per Orthogonal Matching Pursuit (OMP)
    % iteration as long as their corresponding dictionary atoms have a mutual
    % coherence below this threshold.  Default = 0.01
    coh_thresh = 0.01;
    
    % Threshold to consider a Gram matrix negligible
    %
    % Solving for the optimal values of the spike features requires inverting a
    % Gram matrix (dot products between whitened spike waveforms at the selected
    % spike times). This matrix has a banded structure and the cost of inverting
    % it scales with its bandwidth. The whitening filter often makes it so that
    % the dot product between spikes separated by a few ms is negligibly small
    % but still nonzero, and setting these to zero reduces the matrix bandwidth
    % without substantially affecting the solution. This is controlled by this
    % parameter. Default = 0.001
    % See also spkdec.Gramians.getGramSeq
    gram_thresh = 0.001;
    
    % Print status updates to stdout
    %
    % We have a range of verbosity levels:
    %   0,false  No output
    %   1,true   Summary at the start and end of the call to solve()
    %   2        Update every iteration
    % Default = false
    verbose = false;
end

properties (Constant, Hidden)
    errid_arg = 'spkdec:Solver:BadArg';
    errid_dim = 'spkdec:Solver:DimMismatch';
end


% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------


methods
    function obj = Solver(varargin)
        % Solver constructor
        %   obj = Solver(...)
        %
        % Any additional arguments (key/value pairs) will be used to set
        % object properties.
        ip = inputParser();
        ip.KeepUnmatched = true; ip.PartialMatching = false;
        ip.parse( varargin{:} );
        prm = ip.Unmatched;
        for fn = fieldnames(prm)', obj.(fn{1}) = prm.(fn{1}); end
    end
    
    % The main solve() method
    [spk, resid] = solve(self, A, b);
    
    % A wrapper for solve() that performs some postprocessing
    [spk, lims, resid] = detect(self, basis, data, varargin);
end

% ----------------------     Copy and serialization     ------------------------

methods
    function s = saveobj(self)
        s = struct();
        for fn = {'det_refrac','coh_thresh','gram_thresh','verbose'}
            s.(fn{1}) = self.(fn{1});
        end
    end
end
methods (Static)
    function obj = loadobj(s)
        obj = spkdec.Solver(s);
    end
end


% ------------------------------------------------------------------------------
% ====================    Protected properties/methods     =====================
% ------------------------------------------------------------------------------


% Individual steps of the optimization -----------------------------------------

methods (Access=protected)
    % Evaluate the improvement in error from adding a spike
    delta = eval_improvement(self, resid);
    
    % Select new spike times
    new_spk = select_spikes(self, delta, old_spk);
    
    % Solve the linear inverse problem to find spike features
    spk_X = find_spike_features(self, spk);
    
    % Compute the residual
    resid = compute_residual(self, spk);
end

% Temporary cache of problem constants -----------------------------------------

properties (Access=protected)
    A           % Kernels for convolution (SpikeBasis object)
    b           % [T+V x C] Whitened data
    At_b        % [T x K x R x C] output of A.convT(b)
    beta        % Regularizer cost per spike
    select_dt   % Refractory period for spike selection (based on coh_thresh)
    t_start     % tic() when we started this problem
end
methods (Access=protected)
    % Initialization
    consts_init(self, A, b, beta);
    % Cleanup
    function consts_cleanup(self)
        self.A = []; self.b = []; self.At_b = []; self.beta = [];
        self.select_dt = []; self.t_start = [];
    end
end

% Verbose output ---------------------------------------------------------------

properties (Access=protected, Transient)
    vb_format   % Format string for iteration-level output
    last_err    % Previous squared reconstruction error
    last_nnz    % Previous spike count
end
methods (Access=protected)
    % Initialization
    verbose_init(self);
    % Per-iteration udpates
    verbose_update(self, iter, spk, resid);
    % Final update and cleanup
    verbose_cleanup(self, n_iter, spk, resid);
end

end
