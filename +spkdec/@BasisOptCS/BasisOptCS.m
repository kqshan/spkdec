% BasisOptimizer subclass that constrains the bases to be channel-specific (CS)
%
% BasisOptCS properties:
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
% BasisOptCS methods:
%   BasisOptCS  - Construct a new BasisOptCS object
%   optimize    - Optimize the spike basis for the given data
% High-level operations
%   makeBasis   - Construct a new SpikeBasisCS optimized for the given spikes
%   updateBasis - Update a spike basis using proximal gradient descent
% Object management
%   copy        - Create a deep copy of this handle object
%   saveobj     - Serialize a BasisOptCS object to struct
%   loadobj     - [Static] Deserialize a BasisOptCS object from a struct
%
% This class uses the same objective function as its superclass BasisOptimizer,
%   f(basis) = ||data - basis*spikes||^2,
% but it adds an additional constraint that the basis is channel-specific. Since
% this may not allow for fully orthonormal basis vectors, that constraint is
% relaxed to being orthonormal within each channel. The resulting optimization
% problem is thus
%     minimize    f(basis) + lambda*||basis-basis_prev||^2
%   subject to    basis is channel-specific
%                 basis is channelwise orthonormal


classdef BasisOptCS < spkdec.BasisOptimizer

% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------

methods
    function obj = BasisOptCS(varargin)
        % BasisOptCS constructor
        %   obj = BasisOptCS( ... )
        %
        % All input arguments are forwarded to the superclass constructor.
        % See also: spkdec.BasisOptimizer/BasisOptimizer
        obj = obj@spkdec.BasisOptimizer(varargin{:});
    end
    
    % Main optimization routine
    [basis, spk, resid] = optimize(self, spikes, varargin);
    
    % High-level operations
    [basis, spk] = makeBasis(self, spikes, D, varargin);
    [basis, spk] = updateBasis(self, basis, spk, resid, varargin);
end

% ----------------------     Copy and serialization     ------------------------

methods (Static)
    function obj = loadobj(s)
        obj = spkdec.BasisOptCS(s.whbasis, rmfield(s,'whbasis'));
    end
end


% ------------------------------------------------------------------------------
% ====================    Protected properties/methods     =====================
% ------------------------------------------------------------------------------


% Individual steps of the optimization -----------------------------------------

% Note that A and A0 are now [L x D/C x C] matrices in Q2 coordinates instead of
% [L*C x D] matrices in Q1 coordinates.

methods (Access=protected)
    % Modified to represent the basis in Q2 coordinates instead of Q1
    A0 = init_spkbasis(self, D);
    A0 = convert_spkbasis_to_A(self, basis);
    basis = convert_A_to_spkbasis(self, A);
    A = append_bases(self, A1, A2);
    
    % Modified to transform between Q2 and Q1 coordinates
    Ar = get_shifted_basis(self, A, r);
    grad = compute_gradient(self, A, X);
    
    % Modified for new constraints (channel-specific + channelwise-orthonormal)
    A = prox_grad_step(self, A, grad);
end

% Backtracking -----------------------------------------------------------------

methods (Access=protected)
    function lipschitz_init(self, X)
        lipschitz_init@spkdec.BasisOptimizer(self, X);
        % Incorporate the operator norm of map_21 into lip_max
        lbda = self.lambda;
        norm_XXt = self.lip_max - lbda;     % As defined in superclass method
        norm_map21 = norm(self.whbasis.map_21(:,:));
        self.lip_max = norm_map21^2*norm_XXt + lbda;
        % Re-evaluate the starting Lipschitz estimate
        self.lip = (self.lip_max+self.lip_min)/2;
    end
end

end
