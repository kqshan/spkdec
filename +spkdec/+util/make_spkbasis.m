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
%   basis_mode  Spike basis mode: {'channel-specific',['omni-channel']}
%   n_iter      Number of gradient descent iterations           [ 12 ]
%   D_start     # of spike basis waveforms for initialization   [ auto ]
%   make_ind    Rotate basis so features are independent        [ true ]
%   D           Number of spike basis waveforms                 [ 8 ]
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
% If basis_mode=='channel-specific', then `basis` will be a SpikeBasisCS (a
% subclass of SpikeBasis with channel-specific basis waveforms).
%
% See also: spkdec.util.init_spkbasis, spkdec.util.update_spkbasis

%% Deal with inputs

% Optional parameters
ip = inputParser();
is_s_ora = @(x,class) isstruct(x) || isa(x,class);
ip.addParameter('solver',    struct(), @(x) is_s_ora(x,'spkdec.Solver'));
ip.addParameter('optimizer', struct(), @(x) is_s_ora(x,'spkdec.SpikeOptimizer'));
ip.addParameter('basis_mode', 'omni-channel', @ischar);
ip.addParameter('n_iter',     12, @isscalar);
ip.addParameter('D_start',    [], @(x) isempty(x) || isscalar(x));
ip.addParameter('make_ind', true, @isscalar);
ip.addParameter('K',NaN, @isscalar); % Deprecated parameter
ip.addParameter('D',  8, @isscalar);
ip.addParameter('R',  3, @isscalar);
ip.addParameter('L', 25, @isscalar);
ip.addParameter('t0', 9, @isscalar);
ip.addParameter('batch_size', 256*1024, @isscalar);
ip.addParameter('n_batch',           8, @isscalar);
ip.addParameter('verbose',       false, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Special handling for this now-deprecated 'K' parameter
if ~isnan(prm.K)
    warning('The "K" parameter is deprecated and should not be used');
    % The old behavior was to default to a basis_mode='channel-specific'
    if ismember('basis_mode',ip.UsingDefaults)
        prm.basis_mode = 'channel-specific';
    end
    assert(ismember('D',ip.UsingDefaults), ...
        'The "K" and "D" parameter cannot both be given');
    prm.D = prm.K * whitener.C;
end

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

% Let's get some other dimensions and parameters
C = optimizer.C;
basis_mode = prm.basis_mode;

% Rather than initialize them all at once, it seems to be better if we add the
% waveforms a few at a time. Let's determine the schedule.
D_tgt = prm.D;
D_start = prm.D_start;
if strcmp(basis_mode, 'channel-specific')
    % D must always be a multiple of C
    assert(mod(D_tgt,C)==0, ...
        'In the "channel-specific" basis_mode, D must be divisible by C');
    K_tgt = D_tgt/C;
    if isempty(D_start), D_start = C; end
    D_sched = (2:K_tgt)' * C;
else
    % Start with (at most) C waveforms, then step up over (at most) 3 iterations
    if isempty(D_start), D_start = min(D_tgt,C); end
    n_iter = min(3, D_tgt-D_start);
    D_sched = round((1:n_iter)' * (D_tgt-D_start)/n_iter) + D_start;
end

% Initialize using traditional spike detection
[basis, spk] = spkdec.util.init_spkbasis(optimizer, src, D_start, ...
    't0',prm.t0, 'basis_mode',basis_mode, ibatch_prm);
% Report the results
if verbose
    note = sprintf(['Initialized to D=%d using ' ...
        'traditionally-detected spikes'],basis.D);
    fprintf(vb_fmt, toc(t_start), spk.N, spk.N/ibatch_prm.n_batch, note);
end 

% And then add on additional basis waveforms
for D_curr = D_sched'
    % Adjust the spike detection threshold because the D is different
    % We'll set it so to maintain a constant false positive rate (under a
    % chi-squared distribution, an assumption that is definitely wrong).
    det_thresh_tgt = solver.det_thresh;
    log_ccdf_tgt = log(chi2cdf(D_tgt*det_thresh_tgt, D_tgt, 'upper'));
    det_thresh = fzero(@(x) ...
        log(chi2cdf(D_curr*x,D_curr,'upper')) - log_ccdf_tgt, det_thresh_tgt);
    solver2 = copy(solver);
    solver2.det_thresh = det_thresh;
    % Perform the update
    [basis, spk] = spkdec.util.update_spkbasis(basis, src, 'solver',solver2, ...
        'optimizer',optimizer, 'reg_wt',0, 'D_add',D_curr-basis.D, ibatch_prm);
    % Report the results
    if verbose
        note = sprintf(['Increased to D=%d using deconvolution-' ...
            'detected spikes'],basis.D);
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
    if isa(basis, 'spkdec.SpikeBasisCS')
        % Do this per channel
        K = basis.K; C = basis.C;
        basis_cs = basis.basis_cs;
        for c = 1:C
            kk = (1:K)' + K*(c-1);
            basis_cs(:,:,c) = rotate_basis_canonically( ...
                basis_cs(:,:,c), spk.X(kk,:) );
        end
        basis = basis.copy_modifyCS(basis_cs);
    else
        % Do this for all the channels together
        [L,C,D] = size(basis.basis);
        basis_mat = reshape(basis.basis, [L*C, D]);
        basis_mat = rotate_basis_canonically(basis_mat, spk.X);
        basis_mat = reshape(basis_mat, [L C D]);
        basis = basis.copy_modify(basis_mat);
    end
end

% Report the total elapsed time
if verbose
    fprintf('util.make_spkbasis() completed in %.1f sec\n',toc(t_start));
end

end


% --------------------------     Helper functions     --------------------------


function [B, Y] = rotate_basis_canonically(A,X)
% Rotate the given basis so that the features obey some desired properties
%   [B, Y] = rotate_basis_canonically(A,X)
%
% Returns:
%   B       [L x D] rotated basis (B*Y == A*X)
%   Y       [D x N] rotated features (rows of Y are orthogonal)
% Required arguments:
%   A       [L x D] basis
%   X       [D x N] features
%
% Additionally, the rows of Y are sorted in order of decreasing norm, and the
% signs of B and Y are flipped so that:
% * Y(1,:) is generally positive
% * The dot products of B(:,2:end) with B(1,:) are positive
[U,~,~] = svd(gather(double(X*X')), 'econ');
B = A * U;
Y = U' * X;
% Flip the sign of the first basis waveform so that its corresponding feature
% dimension is generally positive
sign_flip = sign(median(Y(1,:)));
sign_flip = gather(double(sign_flip));
if (sign_flip == 0), sign_flip = 1; end
B(:,1) = B(:,1) * sign_flip;
Y(1,:) = Y(1,:) * sign_flip;
% Flip the sign of subsequent waveforms so that the have a positive dot product
% with the first basis waveform
sign_flip = sign(B' * B(:,1));
sign_flip(sign_flip==0) = 1;
B = B .* sign_flip';
Y = Y .* sign_flip;
end
