function basis = make_spkbasis(src, whitener, varargin)
% Create a SpikeBasis object for spike detection in the given source
%   basis = make_spkbasis(src, ...)
%
% Returns:
%   basis       SpikeBasis object fitted to the given source
% Required arguments:
%   src         [Inf x C] DataSrc object to read raw data from
%   whitener    Whitener object for measuring approximation error
% Optional parameters (key/value pairs) [default]:
%   solver      Solver object or params for construction        [ auto ]
%   optimizer   SpikeOptimizer or params for construction       [ auto ]
%   n_iter      Number of gradient descent iterations           [ 12 ]
%   make_ind    Rotate basis so features are independent        [ true ]
%   K           Number of spike basis waveforms per channel     [ 3 ]
%   R           Sub-sample interpolation ratio (1 = no interp)  [ 3 ]
%   L           Spike basis length (#samples)                   [ 25 ]
%   t0          Sample index (1..L) of spike center (t=0)       [ 9 ]
%   batch_size  Size (#samples) of each batch                   [ 256k ]
%   n_batch     Number of randomly-selected batches per iter    [ 8 ]
%   verbose     Print status updates to stdout                  [ false ]
%
% This initializes the spike basis waveforms using spkdec.util.init_spkbasis,
% then performs stochastic gradient descent using spkdec.util.update_spkbasis.
%
% See also: spkdec.util.init_spkbasis, spkdec.util.update_spkbasis

%% Deal with inputs

% Optional parameters
ip = inputParser();
is_s_ora = @(x,class) isstruct(x) || isa(x,class);
ip.addParameter('solver',    struct(), @(x) is_s_ora(x,'spkdec.Solver'));
ip.addParameter('optimizer', struct(), @(x) is_s_ora(x,'spkdec.SpikeOptimizer'));
ip.addParameter('n_iter',     12, @isscalar);
ip.addParameter('make_ind', true, @isscalar);
ip.addParameter('K',  3, @isscalar);
ip.addParameter('R',  3, @isscalar);
ip.addParameter('L', 25, @isscalar);
ip.addParameter('t0', 9, @isscalar);
ip.addParameter('batch_size', 256*1024, @isscalar);
ip.addParameter('n_batch',           8, @isscalar);
ip.addParameter('verbose',       false, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Default optimizer
optimizer = prm.optimizer;
if isstruct(optimizer)
    interp = spkdec.Interpolator.make_interp(prm.L, prm.R);
    optimizer = spkdec.SpikeOptimizer(whitener, 'interp',interp, optimizer);
end

% Default solver
solver = prm.solver;
if isstruct(solver)
    L = optimizer.L;
    solver = spkdec.Solver('det_refrac',ceil(L/2), solver);
end

% Verbose start
verbose = prm.verbose;
if verbose
    fprintf('util.make_spkbasis() started at %s\n', datestr(now(),31));
    fprintf('t_elapsed | #spike /batch | Notes\n');
    vb_fmt = '%9.1f | %6d %6.1f | %s\n';
end
t_start = tic();

%% Initialize basis waveforms

% Perform these steps with a larger number of batches, since the initialization
% steps are bit more sensitive to sampling noise
init_batch_mult = 4;
batch_prm = rmfield(prm, setdiff(fieldnames(prm),{'batch_size','n_batch'}));
ibatch_prm = batch_prm;
ibatch_prm.n_batch = batch_prm.n_batch * init_batch_mult;

% Rather than initialize them all at once, it seems to be better if we add the
% waveforms one at a time. So let's start with K = 1
[basis, spk] = spkdec.util.init_spkbasis(optimizer, src, 1, ...
    't0',prm.t0, ibatch_prm);
% Report the results
if verbose
    fprintf(vb_fmt, toc(t_start), spk.N, spk.N/ibatch_prm.n_batch, ...
        'Initialized to K=1 using PCA of traditionally-detected spikes');
end 

% And then add on additional basis waveforms one by one
for K = 2:prm.K
    % Adjust the spike detection threshold because the K is different
    % We'll set it so to maintain a constant false positive rate (under a
    % chi-squared distribution, an assumption that is definitely wrong).
    KC_tgt = prm.K * basis.C;
    det_thresh_tgt = solver.det_thresh;
    log_ccdf_tgt = log(chi2cdf(KC_tgt*det_thresh_tgt, KC_tgt, 'upper'));
    KC_curr = K * basis.C;
    det_thresh = fzero(@(x) ...
        log(chi2cdf(KC_curr*x,KC_curr,'upper')) - log_ccdf_tgt, det_thresh_tgt);
    solver2 = copy(solver);
    solver2.det_thresh = det_thresh;
    % Perform the update
    [basis, spk] = spkdec.util.update_spkbasis(basis, src, 'solver',solver2, ...
        'optimizer',optimizer, 'reg_wt',0, 'K_add',1, ibatch_prm);
    % Report the results
    if verbose
        note = sprintf(['Increased to K=%d using PCA of deconvolution-' ...
            'detected spikes'],basis.K);
        fprintf(vb_fmt, toc(t_start), spk.N, spk.N/ibatch_prm.n_batch, note);
    end
end

%% Stochastic gradient descent on the basis waveforms

% Perform the descent iterations
n_iter = prm.n_iter;
for iter = 1:n_iter
    % Increase the regularizer weight over time
    reg_wt = (init_batch_mult + iter-1) / (init_batch_mult + n_iter);
    [basis, spk] = spkdec.util.update_spkbasis(basis, src, 'solver',solver, ...
        'optimizer',optimizer, 'reg_wt',reg_wt, batch_prm);
    % Report the results
    if verbose
        if iter==1, note = 'Stochastic gradient descent, '; else, note = ''; end
        note = sprintf('%sreg_wt=%.2f', note, reg_wt);
        fprintf(vb_fmt, toc(t_start), spk.N, spk.N/batch_prm.n_batch, note);
    end
end

% Spike detection is indifferent to orthogonal transformations of the basis
% waveforms within a channel, and it can be convenient to rotate them so that
% the detected features (within a channel) are statistically independent.
if prm.make_ind
    % Separate out the features by channel
    K = basis.K; C = basis.C; N = spk.N;
    spk_X = reshape(spk.X, [K C N]);
    spk_X = permute(spk_X, [1 3 2]); % [K x N x C]
    spk_X = gather(double(spk_X));
    % Perform the rotations on each channel independently
    basis_new = basis.basis;    % [L x K x C]
    for c = 1:C
        % Reconstructed spikes = A*X = A*U*S*V'. Let us define
        %       A2 = A * U      X2 = S * V'
        % Then we still have A2*X2 == A*X, but now the rows of X2 are orthogonal
        % and sorted in descending order of 2-norm.
        [U,~,V] = svd(spk_X(:,:,c), 'econ');
        A2 = basis.basis(:,:,c) * U;
        % For sign consistency, let's flip the sign of the first basis waveform
        % so that its corresponding feature dimension is generally positive.
        % Note that S > 0, so sign(X2) == sign(V').
        A_sign = sign(median(V(:,1)));
        if A_sign == 0, A_sign = 1; end
        A2(:,1) = A2(:,1) * A_sign;
        % And we'll set the signs of the remaining basis waveforms so that they
        % have a positive dot product with the first basis waveform (they may be
        % orthogonal in whitened space, but these dot products are in
        % non-whitened space).
        A_sign = sign(sum(A2 .* A2(:,1),1));
        A_sign(A_sign==0) = 1;
        A2 = A2 .* A_sign;
        % Done with this channel
        basis_new(:,:,c) = A2;
    end
    % Create a new SpikeBasis object with this rotated basis
    basis = basis.copy_modify(basis_new); 
end

% Report the total elapsed time
if verbose
    fprintf('util.make_spkbasis() completed in %.1f sec\n',toc(t_start));
end

end
