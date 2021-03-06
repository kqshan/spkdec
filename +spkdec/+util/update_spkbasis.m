function [basis_new, spk_new] = update_spkbasis(basis, src, varargin)
% Update the spike basis by detecting spikes and minimizing reconstruction error
%   [basis, spk] = update_spkbasis(basis, src, ...)
%
% Returns:
%   basis       New SpikeBasis with updated basis waveforms
%   spk         Detected spikes (Spikes object)
% Required arguments:
%   basis       Original spike basis waveforms (SpikeBasis object)
%   data        [Inf x C] DataSrc object to read raw data from
% Optional parameters (key/value pairs) [default]:
%   solver      Solver object to detect spikes with         [ auto ]
%   optimizer   BasisOptimizer object to use                [ auto ]
%   reg_wt      Relative weight on proximal regularizer     [ 0.1 ]
%   D_add       Number of basis waveforms to add            [ 0 ]
%   batch_size  Size (#samples) of each batch               [ 256k ]
%   n_batch     Number of randomly-selected batches         [ 32 ]
%   verbose     Print status updates to stdout              [ false ]
%   ...         Add'l parameters are forwarded to optimizer.updateBasis()
%
% This randomly selects some batches of data from the given source, detects
% spikes using the given <basis> and <solver>, then uses the given <optimizer>
% to minimize the spike residuals.
%
% reg_wt controls the relative weight of the proximal regularizer, an extra term
% in the optimization that penalizes large changes in the spike basis. Using
% this, update_spkbasis() can be used to implement stochastic gradient descent.
%
% D_add can be used to increase the number of basis waveforms. This may produce
% more stable results than simply calling init_spkbasis() with a larger D.
%
% If the given basis is a SpikeBasisCS (a subclass of SpikeBasis with channel-
% specific basis waveforms), then the output basis will also be a SpikeBasisCS.

%% Deal with inputs

errid_pfx = 'spkdec:util:update_spkbasis';
errid_arg = [errid_pfx ':BadArg'];
errid_dim = [errid_pfx ':DimMismatch'];

% Optional parameters
ip = inputParser();
ip.KeepUnmatched = true; ip.PartialMatching = false;
isemptyora = @(x,class) isempty(x) || isa(x,class);
ip.addParameter('solver',    [], @(x) isemptyora(x,'spkdec.Solver'));
ip.addParameter('optimizer', [], @(x) isemptyora(x,'spkdec.BasisOptimizer'));
ip.addParameter('reg_wt',         0.1, @isscalar);
ip.addParameter('D_add',            0, @isscalar);
ip.addParameter('batch_size',256*1024, @isscalar);
ip.addParameter('n_batch',         32, @isscalar);
ip.addParameter('verbose',      false, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;
addl_args = ip.Unmatched;

% Default solver
solver = prm.solver;
if isempty(solver)
    solver = spkdec.Solver();
end

% Default spike basis optimizer
optimizer = prm.optimizer;
if isempty(optimizer)
    % Create an optimizer that matches the basis
    whbasis = basis.toWhBasis();
    optimizer = spkdec.BasisOptimizer(whbasis);
else
    % Make sure that it matches the basis
    obj_equal = @(a,b) (a==b) || isequal(a.saveobj(), b.saveobj());
    assert(obj_equal(optimizer.whbasis.whitener, basis.whitener) ...
        && obj_equal(optimizer.whbasis.interp, basis.interp), errid_arg, ...
        ['The given BasisOptimizer must use the same whitener and interp' ...
        'olator as the SpikeBasis']);
    % Produce a warning if we're "downgrading" the basis
    if isa(basis,'spkdec.SpikeBasisCS') && ~isa(optimizer,'spkdec.BasisOptCS')
        warning([errid_pfx ':BasisDowngrade'], ['The basis is being updated '...
            'is channel-specific, but the optimizer\nis not, and therefore ' ...
            'the updated basis will no longer be channel-specific.']);
    end
end

% Get some dimensions and local variables
C = basis.C; L = basis.L; W = basis.W;
assert(src.hasShape([Inf C]), errid_dim, 'src.shape must be [Inf x C]');
Lw = L + W - 1;
whitener = basis.whitener;
filt_delay = whitener.delay;
spk_ctr_offset = filt_delay + (basis.t0 - 1);
verbose = prm.verbose;

%% Detect spikes in batches

% Plan the batches
[batch_starts, batch_len] = src.planRand( ...
    'batch_size',prm.batch_size, 'n_batch',prm.n_batch);
assert(batch_len > 12*Lw, errid_arg, ...
    'Batch size is too short for spike detection');
nBatch = length(batch_starts);

% Detect spikes
if verbose
    fprintf('Detecting spikes');
    if solver.verbose, fprintf('%s\n', repmat('-',[1 40])); end
end
spk_all = cell(nBatch,1);
resid_all = cell(nBatch,1);
N_all = zeros(nBatch,1); T_all = zeros(nBatch,1);
for ii = 1:nBatch
    % Read and whiten the data
    x = src.read(batch_starts(ii), batch_len);
    x = whitener.whiten(x);
    data_offset = batch_starts(ii)-1 + filt_delay;
    
    % Perform spike detection
    [spk, lims, resid] = solver.detect(basis, x, ...
        'trunc_1',true, 'trunc_2',true, 'residuals',{'spk'});
    
    % Move to host memory
    spk.setFeat(gather(spk.X));
    resid = gather(resid);
    
    % Store the results
    spk.shiftTimes(data_offset + spk_ctr_offset);
    spk_all{ii} = spk;
    resid_all{ii} = resid.spk;
    % And some stats
    if verbose && ~solver.verbose, fprintf('.'); end
    N_all(ii) = spk.N; T_all(ii) = diff(lims)+1;
end
if verbose
    if solver.verbose, fprintf('%s', repmat('-',[1 40+16])); end
    fprintf('\n');
end

% Concatenate them
spk_all = concat(spk_all{:});
resid_all = cat(3, resid_all{:});  % [Lw x C x N]
if verbose
    fprintf('Detected %d spikes (%.1f +/- %.1f per sec if Fs=25kHz)\n', ...
        spk_all.N, 25e3*sum(N_all)/sum(T_all), 25e3*std(N_all./T_all));
end

%% Update the spike waveforms

if verbose, fprintf('Optimizing spike basis waveforms...\n'); end

[basis_new, spk_new] = optimizer.updateBasis(basis, spk_all, resid_all, ...
    'reg_wt',prm.reg_wt, 'D_add',prm.D_add, addl_args);

end
