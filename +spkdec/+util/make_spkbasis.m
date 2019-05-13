function basis = make_spkbasis(src, whitener, varargin)
% Create a SpikeBasis object for spike detection in the given source
%   basis = make_spkbasis(src, ...)
%
% Returns:
%   basis       SpikeBasis object fitted to the given source
% Required arguments:
%   src         DataSrc object to read from
%   whitener    Whitener object for measuring approximation error
% Optional parameters (key/value pairs) [default]:
%   solver      Solver object for spike detection               [ auto ]
%   optimizer   SpikeOptimizer object for basis fitting         [ auto ]
%   n_iter      Number of gradient descent iterations           [ 12 ]
%   K           Number of spike basis waveforms per channel     [ 3 ]
%   R           Sub-sample interpolation ratio (1 = no interp)  [ 3 ]
%   L           Spike basis length (#samples)                   [ 25 ]
%   t0          Sample index (1..L) of spike center (t=0)       [ 9 ]
%   strategy    Struct defining the fitting strategy            [ auto ]
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
isemptyora = @(x,class) isempty(x) || isa(x,class);
ip.addParameter('solver', [], @(x) isemptyora('spkdec.Solver'));
ip.addParameter('optimizer', [], @(x) isemptyora('spkdec.SpikeOptimizer'));
ip.addParameter('n_iter', 20, @isscalar);
ip.addParameter('K', 3, @isscalar);
ip.addParameter('R', 3, @isscalar);
ip.addParameter('L', 25, @isscalar);
ip.addParameter('t0', 9, @isscalar);
ip.addParameter('batch_size', 256*1024, @isscalar);
ip.addParameter('n_batch', 10, @isscalar);
ip.addParameter('verbose', false, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;
verbose = prm.verbose;

% Default optimizer
optimizer = prm.optimizer;
if isempty(optimizer)
    interp = spkdec.Interpolator.make_interp(prm.L, prm.R);
    whbasis = spkdec.WhitenerBasis(whitener, 'interp',interp);
    optimizer = spkdec.SpikeOptimizer(whbasis);
end

% Default solver
solver = prm.solver;
if isempty(solver)
    L = optimizer.L;
    solver = spkdec.Solver('det_refrac',ceil(L/2));
end

%% Initialize basis waveforms

if verbose, fprintf('Initializing basis waveforms'); end
t_start = tic();

% Perform these steps with a larger number of batches, since the initialization
% steps are bit more sensitive to sampling noise
init_batch_mult = 4;
batch_prm = rmfield(prm, setdiff(fieldnames(prm),{'batch_size','n_batch'}));
ibatch_prm = batch_prm;
ibatch_prm.n_batch = batch_prm.n_batch * init_batch_mult;

% Rather than initialize them all at once, it seems to be better if we add the
% waveforms one at a time. So let's start with K = 1
basis = spkdec.util.init_spkbasis(optimizer, src, 1, 't0',prm.t0, ibatch_prm);
if verbose, fprintf('.'); end

% And then add on additional basis waveforms one by one
for K = 2:prm.K
    basis = spkdec.util.update_spkbasis(basis, src, 'solver',solver, ...
        'optimizer',optimizer, 'reg_wt',0, 'K_add',1, ibatch_prm);
    if verbose, fprintf('.'); end
end
if verbose, fprintf('Done in %.1f sec\n',toc(t_start)); end

%% Stochastic gradient descent on the basis waveforms

if verbose, fprintf('Performing gradient descent'); end
t_start = tic();

% Perform the descent iterations
n_iter = prm.n_iter;
for iter = 1:n_iter
    reg_wt = (init_batch_mult + iter-1) / (init_batch_mult + n_iter);
    basis = spkdec.util.update_spkbasis(basis, src, 'solver',solver, ...
        'optimizer',optimizer, 'reg_wt',reg_wt, batch_prm);
    if verbose, fprintf('.'); end
end
if verbose, fprintf('Done in %.1f sec\n',toc(t_start)); end

end
