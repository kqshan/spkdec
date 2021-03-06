function test(varargin)
% Run unit tests on this class
%   SpikeBasis.test( ... )
%
% Optional parameters (key/value pairs) [default]:
%   do_gpu      Run GPU-based tests             [ auto ]
%   nCases      #cases for varying dimensions   [ 10 ]
%   tol         Relative error tolerance        [ 10 ]
%   verbose     Display more output             [ false ]

% Optional params
ip = inputParser();
ip.addParameter('do_gpu', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('benchmark', false, @isscalar);
ip.addParameter('L_max', 100, @isscalar);
ip.addParameter('C_max',  64, @isscalar);
ip.addParameter('D_max', 250, @isscalar);
ip.addParameter('W_max', 500, @isscalar);
ip.addParameter('R_max',   9, @isscalar);
ip.addParameter('CD_max',500, @isscalar);
ip.addParameter('nCases', 10, @(x) isscalar(x) && x >= 6);
ip.addParameter('tol', 10, @isscalar);
ip.addParameter('verbose', false, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;
% Default value for do_gpu
if isempty(prm.do_gpu)
    try
        gpuArray();
        prm.do_gpu = true;
    catch
        prm.do_gpu = false;
    end
end

% Run the individual unit tests
errmsgs = {};
for test_func_cell = localfunctions()'
    test_func = test_func_cell{1};
    % Ignore any local functions that don't start with "test"
    test_str = func2str(test_func);
    if ~startsWith(test_str,'test'), continue; end
    % Skip the performance benchmarks if desired
    if ~prm.benchmark && contains(test_str,'performance'), continue; end
    % Run the test
    fprintf('%s%s', test_str, repmat('.',1,max(3,40-length(test_str))));
    try
        test_func(prm);
        fprintf('OK\n');
    catch mexc
        fprintf('FAIL\n');
        errmsgs{end+1} = sprintf('%s: %s',test_str,mexc.message); %#ok<AGROW>
    end
end

% Throw an error
assert(isempty(errmsgs), 'spkdec:SpikeBasis:TestFail', ...
    sprintf('%s\n', errmsgs{:}));

end


% -----------------------------     Unit tests     -----------------------------


function test_object_management(~)
% Run tests on basic object management (construction, copy, serialization)

% Constructor with arguments
[kernels, whitener, interp] = create_random_basis();
[L,C,D] = size(kernels); t0 = randi(L);
x = spkdec.SpikeBasis(kernels, 'whitener',whitener, 'interp',interp, 't0',t0);
assert(x.L==L && x.C==C && x.D==D);
assert(x.t0==t0 && isequal(x.whitener,whitener) && isequal(x.interp,interp));
% Copy
y = x.copy();
for fn = properties(x)', assert(isequal(x.(fn{1}),y.(fn{1}))); end
% Serialization
s = x.saveobj();
y = spkdec.SpikeBasis.loadobj(s);
for fn = properties(x)', assert(isequal(x.(fn{1}),y.(fn{1}))); end

end



function test_spike_convolution(prm)
% Test the conv_spk() method

% Decide on the cases to try
LCDWR = randomize_case_dims(prm);
option_cases = struct('use_f32',{false; true}, 'use_gpu',false);
if prm.do_gpu
    option_cases = [option_cases; option_cases];
    [option_cases(3:4).use_gpu] = deal(true);
end
% Fix the data length and the number of spikes
T = 1e4;
N = 70;
% Verbose header
verbose = prm.verbose;
if verbose
    fprintf('\n');
    fprintf('  %-16s |     Error/eps\n', '  Dimensions');
    fprintf('  %3s %2s %3s %3s %1s | %6s  %6s\n', ...
        'L','C','D','W','R', 'f64','f32' );
    if prm.do_gpu, fprintf('\b  %6s  %6s\n', 'f64gpu', 'f32gpu'); end
end
% Try each of these cases
errmsg = {};
for ii = 1:prm.nCases
    % Construct the data
    [kernels, whitener, interp] = create_random_basis(LCDWR(ii,:));
    [L,C,D] = size(kernels); W = whitener.W; R = interp.R;
    t0 = randi(L);
    A = spkdec.SpikeBasis(kernels, 'whitener',whitener,'interp',interp,'t0',t0);
    spk_rt = randperm(R*T, N);
    spk_t = ceil(spk_rt/R);
    spk_r = spk_rt - R*(spk_t-1);
    spk_X_ref = randn(D, N);
    if verbose, fprintf('  %3d %2d %3d %3d %1d |',L,C,D,W,R); end
    % Compute the reference result
    y_ref = zeros(T+L-1, C);
    y_spk = reshape(kernels,[L*C,D]) * spk_X_ref;
    y_spk = reshape(y_spk, [L C N]);
    y_spk = interp.shiftArr(y_spk, spk_r);
    for n = 1:N
        offset = spk_t(n) - 1;
        y_ref(offset+(1:L),:) = y_ref(offset+(1:L),:) + y_spk(:,:,n);
    end
    y_ref = whitener.whiten(y_ref, 'bounds','keep');
    y_norm = norm(y_ref,'fro');
    % Try each of our option cases
    for opt = option_cases'
        % Perform the convolution
        spk_X = spkdec.Math.typeconv(spk_X_ref, opt);
        spk = spkdec.Spikes(spk_t, spk_r, spk_X);
        y = A.conv_spk(spk, T);
        % Check the datatype
        case_desc = sprintf(['L=%d, C=%d, D=%d, W=%d, R=%d, use_f32=%d, ' ...
            'use_gpu=%d'], L,C,D,W,R, opt.use_f32, opt.use_gpu);
        assert(spkdec.Math.typecheck(y, opt), ...
            'Unexpected output type in: %s\n', case_desc);
        % Evaluate the error
        err = norm(y - y_ref,'fro') / y_norm;
        if opt.use_f32
            err_rel = err / eps('single');
        else
            err_rel = err / eps('double');
        end
        if verbose, fprintf(' %6.1f ',err_rel); end
        % Log the error if it exceeds our threshold
        if err_rel > prm.tol
            errmsg{end+1} = sprintf(['Excessive error (%.1fx epsilon) ' ...
                'in: %s\n'], err_rel, case_desc); %#ok<AGROW>
        end
    end
    % Go on to the next dimension case
    if verbose, fprintf('\n'); end
end
% Collect the errors
if verbose, fprintf(repmat('_',[1 40])); end
assert(isempty(errmsg), horzcat(errmsg{:}));
end


function test_spike_reconst(prm)
% Test the reconst() method

% Decide on the cases to try
LCDWR = randomize_case_dims(prm);
option_cases = struct('use_f32',{false; true}, 'use_gpu',false);
if prm.do_gpu
    option_cases = [option_cases; option_cases];
    [option_cases(3:4).use_gpu] = deal(true);
end
% Fix the data length the number of spikes
N = 70;
% Verbose header
verbose = prm.verbose;
if verbose
    fprintf('\n');
    fprintf('  %-15s |     Error/eps\n', '  Dimensions');
    fprintf('  %3s %2s %3s %3s %1s | %6s  %6s\n', ...
        'L','C','D','W','R', 'f64','f32' );
    if prm.do_gpu, fprintf('\b  %6s  %6s\n', 'f64gpu', 'f32gpu'); end
end
% Try each of these cases
errmsg = {};
for ii = 1:prm.nCases
    % Construct the data
    [kernels, whitener, interp] = create_random_basis(LCDWR(ii,:));
    [L,C,D] = size(kernels); W = whitener.W; R = interp.R;
    t0 = randi(L);
    A = spkdec.SpikeBasis(kernels, 'whitener',whitener,'interp',interp,'t0',t0);
    spk_r = randi(R, [N 1]);
    spk_X_ref = randn(D, N);
    if verbose, fprintf('  %3d %2d %3d %3d %1d |',L,C,D,W,R); end
    % Compute the reference result
    y_ref = reshape(kernels,[L*C,D]) * spk_X_ref;
    y_ref = interp.shiftArr(reshape(y_ref,[L C N]), spk_r);
    y_ref = whitener.whiten(y_ref, 'bounds','keep');
    % Try each of our option cases
    for opt = option_cases'
        % Perform the convolution
        spk_X = spkdec.Math.typeconv(spk_X_ref, opt);
        spk = spkdec.Spikes(zeros(N,1), spk_r, spk_X);
        y = A.reconst(spk);
        % Check the datatype
        case_desc = sprintf(['L=%d, C=%d, D=%d, W=%d, R=%d, use_f32=%d, ' ...
            'use_gpu=%d: '], L,C,D,W,R, opt.use_f32, opt.use_gpu);
        assert(spkdec.Math.typecheck(y, opt), ...
            'Unexpected output type in: %s\n', case_desc);
        % Evaluate the error
        err = norm(y(:) - y_ref(:)) / norm(y_ref(:));
        if opt.use_f32
            err_rel = err / eps('single');
        else
            err_rel = err / eps('double');
        end
        if verbose, fprintf(' %6.1f ',err_rel); end
        % Log the error if it exceeds our threshold
        if err_rel > prm.tol
            errmsg{end+1} = sprintf(['Excessive error (%.1fx epsilon) ' ...
                'in: %s\n'], err_rel, case_desc); %#ok<AGROW>
        end
    end
    % Go on to the next dimension case
    if verbose, fprintf('\n'); end
end
% Collect the errors
if verbose, fprintf(repmat('_',[1 40])); end
assert(isempty(errmsg), horzcat(errmsg{:}));
end


function test_spike_norms(prm)
% Test the spkNorms() method

% Decide on the cases to try
LCDWR = randomize_case_dims(prm);
option_cases = struct('use_f32',{false; true}, 'use_gpu',false);
if prm.do_gpu
    option_cases = [option_cases; option_cases];
    [option_cases(3:4).use_gpu] = deal(true);
end
% Fix the data length the number of spikes
N = 70;
% Verbose header
verbose = prm.verbose;
if verbose
    fprintf('\n');
    fprintf('  %-15s |     Error/eps\n', '  Dimensions');
    fprintf('  %3s %2s %3s %3s %1s | %6s  %6s\n', ...
        'L','C','D','W','R', 'f64','f32' );
    if prm.do_gpu, fprintf('\b  %6s  %6s\n', 'f64gpu', 'f32gpu'); end
end
% Try each of these cases
errmsg = {};
for ii = 1:prm.nCases
    % Construct the data
    [kernels, whitener, interp] = create_random_basis(LCDWR(ii,:));
    [L,C,D] = size(kernels); W = whitener.W; R = interp.R;
    t0 = randi(L);
    A = spkdec.SpikeBasis(kernels, 'whitener',whitener,'interp',interp,'t0',t0);
    spk_r = randi(R, [N 1]);
    spk_X_ref = randn(D, N);
    if verbose, fprintf('  %3d %2d %3d %3d %1d |',L,C,D,W,R); end
    % Compute the reference result
    spk_ref = spkdec.Spikes(zeros(N,1), spk_r, spk_X_ref);
    spikes = A.reconst(spk_ref);
    y_ref = sqrt(sum(sum(spikes.^2,1),2));
    y_ref = y_ref(:);
    % Try each of our option cases
    for opt = option_cases'
        % Perform the convolution
        spk_X = spkdec.Math.typeconv(spk_X_ref, opt);
        spk = spkdec.Spikes(zeros(N,1), spk_r, spk_X);
        y = A.spkNorms(spk);
        % Check the datatype
        case_desc = sprintf(['L=%d, C=%d, D=%d, W=%d, R=%d, use_f32=%d, ' ...
            'use_gpu=%d: '], L,C,D,W,R, opt.use_f32, opt.use_gpu);
        assert(spkdec.Math.typecheck(y, opt), ...
            'Unexpected output type in: %s\n', case_desc);
        % Evaluate the error
        err = norm(y(:) - y_ref(:)) / norm(y_ref(:));
        if opt.use_f32
            err_rel = err / eps('single');
        else
            err_rel = err / eps('double');
        end
        if verbose, fprintf(' %6.1f ',err_rel); end
        % Log the error if it exceeds our threshold
        if err_rel > prm.tol
            errmsg{end+1} = sprintf(['Excessive error (%.1fx epsilon) ' ...
                'in: %s\n'], err_rel, case_desc); %#ok<AGROW>
        end
    end
    % Go on to the next dimension case
    if verbose, fprintf('\n'); end
end
% Collect the errors
if verbose, fprintf(repmat('_',[1 40])); end
assert(isempty(errmsg), horzcat(errmsg{:}));
end


% ------------------------------     Helpers     -------------------------------


function [kernels, whitener, interp] = create_random_basis(LCDWR)
% Randomly create some problem data
%   [kernels, whitener, interp] = create_random_basis(LKC)
% Returns:
%   kernels     [L x C x D] spike basis waveforms
%   whitener    Whitener object
%   interp      Interpolator object
% Optional arguments [default]:
%   LKCWR       [L, K, C, W, R] dimensions      [ 25 4 12 201 2 ]
if (nargin < 1), LCDWR = [23 4 12 201 2]; end
L = LCDWR(1); C = LCDWR(2); D = LCDWR(3); W = LCDWR(4); R = LCDWR(5);
% Generate the stuff
kernels = randn(L,C,D);
whitener = spkdec.Whitener('wh_filt',randn(W,C), 'wh_ch',randn(C,C), ...
    'delay',randi(W)-1 );
interp_methods = {'linear','hermite','spline','sinc'};
method = interp_methods{randi(length(interp_methods))};
interp = spkdec.Interpolator.make_interp(L, R, 'method',method);
end


function LCDWR = randomize_case_dims(prm)
% Randomly select the dimensions of the test cases to try
%   LCDWR = randomize_case_dims(prm)
%
% Returns:
%   LCDWR   [nCases x 5] dimensions: [L C D W R]
N = prm.nCases;
L = ceil((sqrt(prm.L_max) .* rand(N,1)).^2);
C = ceil((sqrt(prm.C_max) .* rand(N,1)).^2);
D_max = floor(prm.CD_max ./ C);
D_max = min(D_max, prm.D_max);
D = ceil(D_max .* rand(N,1));
W = randi(prm.W_max, [N 1]);
R = randi(prm.R_max, [N 1]);
LCDWR = [L C D W R];
% Use a deterministic one for the first case
LCDWR(1,:) = [23 4 12 201 2];
% And for the next 5 cases, set one of the dimensions to 1 to make sure that we
% can handle edge cases like these
for ii = 1:5
    LCDWR(ii+1,ii) = 1;
end
% We implicitly assume that L > D/C, so don't test any cases that violate this
L = LCDWR(:,1); C = LCDWR(:,2); D = LCDWR(:,3);
LCDWR(:,1) = max(L, ceil((D+1)./C));
end
