function test(varargin)
% Run unit tests on this class
%   SpikeBasisCS.test( ... )
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
ip.addParameter('K_max',  20, @isscalar);
ip.addParameter('C_max',  64, @isscalar);
ip.addParameter('W_max', 500, @isscalar);
ip.addParameter('R_max',   9, @isscalar);
ip.addParameter('KC_max',256, @isscalar);
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
assert(isempty(errmsgs), 'spkdec:SpikeBasisCS:TestFail', ...
    sprintf('%s\n', errmsgs{:}));

end


% -----------------------------     Unit tests     -----------------------------


function test_object_management(~)
% Run tests on basic object management (construction, copy, serialization)

% Constructor with arguments
[kernels, whitener, interp] = create_random_basis();
[L,K,C] = size(kernels); t0 = randi(L);
x = spkdec.SpikeBasisCS(kernels, 'whitener',whitener, 'interp',interp, 't0',t0);
assert(x.L==L && x.C==C && x.K==K);
assert(x.t0==t0 && isequal(x.whitener,whitener) && isequal(x.interp,interp));
% Copy
y = x.copy();
for fn = properties(x)', assert(isequal(x.(fn{1}),y.(fn{1}))); end
% Serialization
s = x.saveobj();
y = spkdec.SpikeBasisCS.loadobj(s);
for fn = properties(x)', assert(isequal(x.(fn{1}),y.(fn{1}))); end

end



function test_spike_convolution(prm)
% Test the conv_spk() method

% Decide on the cases to try
LKCWR = randomize_case_dims(prm);
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
    fprintf('  %-15s |     Error/eps\n', '  Dimensions');
    fprintf('  %3s %2s %2s %3s %1s | %6s  %6s\n', ...
        'L','K','C','W','R', 'f64','f32' );
    if prm.do_gpu, fprintf('\b  %6s  %6s\n', 'f64gpu', 'f32gpu'); end
end
% Try each of these cases
errmsg = {};
for ii = 1:prm.nCases
    % Construct the data
    [kernels, whitener, interp] = create_random_basis(LKCWR(ii,:));
    [L,K,C] = size(kernels); W = whitener.W; R = interp.R;
    t0 = randi(L);
    A = spkdec.SpikeBasisCS(kernels, ...
        'whitener',whitener, 'interp',interp, 't0',t0);
    spk_rt = randperm(R*T, N);
    spk_t = ceil(spk_rt/R);
    spk_r = spk_rt - R*(spk_t-1);
    spk_X_ref = randn(K*C, N);
    if verbose, fprintf('  %3d %2d %2d %3d %1d |',L,K,C,W,R); end
    % Compute the reference result
    y_ref = zeros(T+L-1, C);
    for n = 1:N
        offset = spk_t(n) - 1;
        r = spk_r(n);
        for c = 1:C
            kk = (1:K) + K*(c-1);
            y_spk = kernels(:,:,c) * spk_X_ref(kk,n);
            y_spk = interp.shift(y_spk, r);
            y_ref(offset+(1:L),c) = y_ref(offset+(1:L),c) + y_spk;
        end
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
        case_desc = sprintf(['L=%d, K=%d, C=%d, W=%d, R=%d, use_f32=%d, ' ...
            'use_gpu=%d'], L,K,C,W,R, opt.use_f32, opt.use_gpu);
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
LKCWR = randomize_case_dims(prm);
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
    fprintf('  %3s %2s %2s %3s %1s | %6s  %6s\n', ...
        'L','K','C','W','R', 'f64','f32' );
    if prm.do_gpu, fprintf('\b  %6s  %6s\n', 'f64gpu', 'f32gpu'); end
end
% Try each of these cases
errmsg = {};
for ii = 1:prm.nCases
    % Construct the data
    [kernels, whitener, interp] = create_random_basis(LKCWR(ii,:));
    [L,K,C] = size(kernels); W = whitener.W; R = interp.R;
    t0 = randi(L);
    A = spkdec.SpikeBasisCS(kernels, ...
        'whitener',whitener, 'interp',interp, 't0',t0);
    spk_r = randi(R, [N 1]);
    spk_X_ref = randn(K*C, N);
    if verbose, fprintf('  %3d %2d %2d %3d %1d |',L,K,C,W,R); end
    % Compute the reference result
    y_ref = zeros(L+W-1, C, N);
    for n = 1:N
        y_spk = zeros(L,C);
        for c = 1:C
            kk = (1:K) + K*(c-1);
            y_spk(:,c) = kernels(:,:,c) * spk_X_ref(kk,n);
        end
        y_spk = interp.shift(y_spk, spk_r(n));
        y_ref(:,:,n) = whitener.whiten(y_spk, 'bounds','keep');
    end
    % Try each of our option cases
    for opt = option_cases'
        % Perform the convolution
        spk_X = spkdec.Math.typeconv(spk_X_ref, opt);
        spk = spkdec.Spikes(zeros(N,1), spk_r, spk_X);
        y = A.reconst(spk);
        % Check the datatype
        case_desc = sprintf(['L=%d, K=%d, C=%d, W=%d, R=%d, use_f32=%d, ' ...
            'use_gpu=%d: '], L,K,C,W,R, opt.use_f32, opt.use_gpu);
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
LKCWR = randomize_case_dims(prm);
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
    fprintf('  %3s %2s %2s %3s %1s | %6s  %6s\n', ...
        'L','K','C','W','R', 'f64','f32' );
    if prm.do_gpu, fprintf('\b  %6s  %6s\n', 'f64gpu', 'f32gpu'); end
end
% Try each of these cases
errmsg = {};
for ii = 1:prm.nCases
    % Construct the data
    [kernels, whitener, interp] = create_random_basis(LKCWR(ii,:));
    [L,K,C] = size(kernels); W = whitener.W; R = interp.R;
    t0 = randi(L);
    A = spkdec.SpikeBasisCS(kernels, ...
        'whitener',whitener, 'interp',interp, 't0',t0);
    spk_r = randi(R, [N 1]);
    spk_X_ref = randn(K*C, N);
    if verbose, fprintf('  %3d %2d %2d %3d %1d |',L,K,C,W,R); end
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
        case_desc = sprintf(['L=%d, K=%d, C=%d, W=%d, R=%d, use_f32=%d, ' ...
            'use_gpu=%d: '], L,K,C,W,R, opt.use_f32, opt.use_gpu);
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


function [kernels, whitener, interp] = create_random_basis(LKCWR)
% Randomly create some problem data
%   [kernels, whitener, interp] = create_random_basis(LKC)
% Returns:
%   kernels     [L x K x C] spike basis waveforms
%   whitener    Whitener object
%   interp      Interpolator object
% Optional arguments [default]:
%   LKCWR       [L, K, C, W, R] dimensions      [ 25 3 4 201 2 ]
if (nargin < 1), LKCWR = [23 3 4 201 2]; end
L = LKCWR(1); K = LKCWR(2); C = LKCWR(3); W = LKCWR(4); R = LKCWR(5);
% Generate the stuff
kernels = randn(L,K,C);
whitener = spkdec.Whitener('wh_filt',randn(W,C), 'wh_ch',randn(C,C), ...
    'delay',randi(W)-1 );
interp_methods = {'linear','hermite','spline','sinc'};
method = interp_methods{randi(length(interp_methods))};
interp = spkdec.Interpolator.make_interp(L, R, 'method',method);
end


function LKCWR = randomize_case_dims(prm)
% Randomly select the dimensions of the test cases to try
%   LKCWR = randomize_case_dims(prm)
%
% Returns:
%   LKCWR   [nCases x 5] dimensions: [L K C W R]
N = prm.nCases;
L = ceil((sqrt(prm.L_max) .* rand(N,1)).^2);
C = ceil((sqrt(prm.C_max) .* rand(N,1)).^2);
K_max = floor(prm.KC_max ./ C);
K_max = min(K_max, prm.K_max);
K = ceil(K_max .* rand(N,1));
W = randi(prm.W_max, [N 1]);
R = randi(prm.R_max, [N 1]);
LKCWR = [L K C W R];
% Use a deterministic one for the first case
LKCWR(1,:) = [23 3 4 201 2];
% And for the next 5 cases, set one of the dimensions to 1 to make sure that we
% can handle edge cases like these
for ii = 1:5
    LKCWR(ii+1,ii) = 1;
end
% We implicitly assume that L > K, so don't test any cases that violate this
LKCWR(:,1) = max(LKCWR(:,2)+1, LKCWR(:,1));
end
