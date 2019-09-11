function [basis, spk, resid] = optimize(self, data, varargin)
% Find a spike waveform basis that minimizes the reconstruction error
%   [basis, spk, resid] = optimize(self, data, ...)
%
% Returns:
%   basis       [L x C x D] optimized spike basis waveforms
%   spk         Spikes object (where t is the shift in detected spike time)
%   resid       [L+W-1 x C x N] spike residuals (whitened) after optimization
% Required arguments:
%   data        [L+W-1 x C x N] detected spike waveforms (whitened)
% Optional parameters (key/value pairs) [default]:
%   lambda      Proximal regularizer weight                 [ 0 ]
%   basis_prev  Previous basis (for proximal regularizer)   [ none ]
%   D           Number of basis waveforms overall         [defer to basis_prev]
%   zero_pad    [pre,post] #samples of zero-padding to add  [ 0,0 ]
%   spk_r       [N x 1] sub-sample shift for each spike     [ auto ]
%
% This finds the basis waveforms and spike features to solve:
%     minimize    ||data - basis*spk.X||^2 + lambda*||basis-basis_prev||^2
%   subject to    basis is orthonormal
% where the norms are defined in terms of the whitened inner product.
%
% If self.dt_search > 0 and/or self.whbasis.R > 1 (unless `spk_r` is given),
% then this process will also search over available full- and/or sub-sample
% shifts in the detected spike times. This reduces the incentive to represent
% such shifts using the spike basis itself.


% --------------------     Problem description     ------------------------

% First, let's write the original problem in matrix notation. This will require
% reshaping some of our data:
%   data        [(L+W-1)*C x N] observed whitened spike waveforms
%   wh_00       [(L+W-1)*C x L*C] whitening operation (self.whbasis.wh_00)
%   basis       [L*C x D] spike basis waveforms
%   basis_prev  [L*C x D] previous basis for the regularizer
%   X           [D x N] spikes in feature space
% Then we can pose the original problem as:
%     minimize  ||data-wh_00*basis*X||^2 + lambda*||wh_00*(basis-basis_prev)||^2
%   subject to  basis is orthonormal


% Next, let's see how a change of variables can help us restate this problem.
%
% whbasis.Q1 defines a L*C dimensional orthonormal basis for the span of the
% whitener, which means that the component of <data> that is orthogonal to Q1 is
% forever out of the reach of this optimization. Let's introduce
%   Y       [L*C x N] observed waveforms in Q1 coordinates: Y = Q1' * data
%   A       [L*C x D] whitened spike basis in Q1 coordinates: A = wh_01 * basis
%   A0      [L*C x D] whitened previous basis in Q1 coordinates
% And since wh_00 == Q1 * wh_01 and Q1'*Q1 == I, our problem is equivalent to:
%     minimize  ||Y - A*X||^2 + lambda*||A - A0||^2
%   subject to  basis is orthonormal


% So that's the problem that we are solving here. We first convert the given
% data (<data> and basis_prev) into these Q1 coordinates (Y and A0), solve for
% A, then find <basis> such that A == wh_01 * basis.


% We perform this minimization using alternating descent. Ideally, each
% iteration would consist of two steps:
% 1. Find X that minimizes the objective while holding A constant
% 2. Find A that minimizes the objective (and satisfies the constraints) while
%    holding X constant
%
% However, step 2 is kinda hard, so we replace it with:
% 2. Take a gradient descent step in A, then project this into the feasible set
%    with respect to the constraints on A
% This is also known as proximal gradient descent. The step size is controlled
% by a parameter known as the local Lipchitz estimate (step size = 1 / lip) and
% we want the step size to be as large as possible but not too large. It turns
% out that we can evaluate whether a step was "too large" after the fact, a
% process known as "backtracking", which turns step 2 into:
% 2a. Tentatively increase the step size (decrease the Lipschitz estimate)
% 2b. Take a proximal gradient descent step in A
% 2c. Evaluate whether this step was too large. If so, reduce the step size and
%     go back to step 2b.


% Step 1 is also complicated by a couple features that we now support. As
% stated, it should be fairly simple:
%   X = (A'*map_21'*map_21*A) \ (A'*map_21'*Y)
%
% However, we have allowed for full-sample shifts in the data (via the dt_search
% option). This essentially creates S versions of the data:
%   Y{s} = Q1' * shift(s) * data
% Our optimization of X then searches over these options to find which one
% minimizes the residual (keeping in mind to account for the effect of the shift
% operator itself). It is then this X and Y (on a spike-by-spike basis) that are
% used when computing the gradient for step 2.
%
% We have also allowed for sub-sample shifts via the Interpolator object. This
% cannot be implemented by simply shifting the data, and the relationship
% between these residuals and the gradient is a little more complicated.


% -----------------     Initialize the problem     ------------------------

% Start counting the runtime
self.t_start = tic();

% Optional parameters
ip = inputParser();
ip.addParameter('lambda', 0, @isscalar);
ip.addParameter('basis_prev', []);
ip.addParameter('D', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('zero_pad', [0 0], @(x) numel(x)==2);
ip.addParameter('spk_r', [], @(x) isempty(x) || numel(x)==size(data,3));
ip.parse( varargin{:} );
prm = ip.Results;

% Convert the given spikes into Q1 coordinates
self.Y = self.convert_spikes_to_Y(data, prm.zero_pad);

% Put ourselves into omni-channel (each waveform spans all channels) mode
self.basis_mode = 'omni-channel';

% Get a starting spike basis
L = self.L; C = self.C;
if isempty(prm.basis_prev)
    assert(prm.lambda==0, self.errid_arg, ...
        'basis_prev must be specified if lambda ~= 0');
    assert(~isempty(prm.D), self.errid_arg, ...
        'D must be specified if basis_prev is not given');
    % Initialize this based on the data
    A = self.init_spkbasis(prm.D);
else
    [L_, C_, D] = size(prm.basis_prev);
    assert(L_==L && C_==C, self.errid_dim, ...
        'basis_prev must be [L x C x D] with L=%d and C=%d',L,C);
    % Convert the given spike basis
    A = self.whbasis.wh_01(:,:) * reshape(prm.basis_prev, [L*C, D]);
end
% Store this (and lambda) in our object-level cache
self.A0 = A;
self.lambda = prm.lambda;
% Also save the user-specified spk_r, if given
self.spk_r = prm.spk_r(:);
if ~isempty(self.spk_r) && (self.dt_search > 0)
    warning('spkdec:BasisOptimizer:WeirdSearch', ['The sub-sample shift is ' ...
        'fixed by the user-specified spk_r,\nbut we are still searching '...
        'over full-sample shifts since dt_search > 0. This is kinda weird']);
end

% Start the verbose output
self.verbose_init();


% ---------------     Perform alternating minimization     ----------------


for iter = 1:self.n_iter
    % Optimize the spikes with basis held constant
    X = self.optimize_spk(A);
    
    % Deal with the local Lipschitz estimate
    if (iter==1)
        % Couldn't initialize these outside the loop because of dependence on X
        self.lipschitz_init(X);
        self.verbose_update(0, A, X);
    else
        % Attempt to increase the step size every iteration
        self.lipschitz_decay();
    end
    
    % Perform a proximal gradient descent step on A
    prev_A = A;
    grad = self.compute_gradient(prev_A, X);
    step_ok = false;
    while ~step_ok
        A = self.prox_grad_step(prev_A, grad);
        % Determine if we need to backtrack
        step_ok =self.eval_step(A, prev_A, X);
        if ~step_ok, self.lipschitz_backtrack(); end
    end
    
    % Verbose update
    self.verbose_update(iter, A, X);
end


% --------------------------    Finish up     -----------------------------


% Convert the basis back into raw waveforms
basis = self.convert_A_to_spkbasis(A);

% Construct the Spikes object
spk_t = X.s - (self.dt_search+1);
spk = spkdec.Spikes(spk_t, X.r, X.X);

% Compute the residual
if nargout >= 3
    resid = self.eval_error(A, X);  % [L*C x N] in Q1 coordinates
    Lw = L + self.W-1;
    resid = reshape(self.whbasis.Q1,[Lw*C,L*C]) * resid;
    resid = reshape(resid, [Lw, C, spk.N]);
end

% Cleanup
self.verbose_cleanup();
self.lipschitz_cleanup();
self.spk_r = [];
self.t_start = []; self.Y = []; self.A0 = []; self.lambda = [];
self.basis_mode = [];

end
