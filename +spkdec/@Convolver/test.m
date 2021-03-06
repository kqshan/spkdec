function test(varargin)
% Run unit tests on this class
%   Convolver.test( ... )
%
% Optional parameters (key/value pairs) [default]:
%   do_gpu      Run GPU-based tests             [ auto ]
%   benchmark   Run performance tests           [ false ]
%   nCases      #cases for varying dimensions   [ 10 ]
%   tol         Relative error tolerance        [ 10 ]
%   verbose     Display more output             [ false ]

% Optional params
ip = inputParser();
ip.addParameter('do_gpu', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('benchmark', false, @isscalar);
ip.addParameter('L_max', 600, @isscalar);
ip.addParameter('C_max',  16, @isscalar);
ip.addParameter('D_max',  50, @isscalar);
ip.addParameter('CDT_max', 1e6, @isscalar);
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
assert(isempty(errmsgs), 'spkdec:Convolver:TestFail', ...
    sprintf('%s\n', errmsgs{:}));

end


% -----------------------------     Unit tests     -----------------------------


function test_object_management(~)
% Run tests on basic object management (construction, copy, serialization)

% Constructor with arguments
kernels = create_random_kernels();
[L,C,D] = size(kernels); t0 = randi(L);
x = spkdec.Convolver(kernels, 't0',t0);
assert(isequal(x.kernels,kernels) && isequal(x.t0,t0));
assert(x.L==L && x.C==C && x.D==D);
% Copy
y = x.copy();
for fn = properties(x)', assert(isequal(x.(fn{1}),y.(fn{1}))); end
% Serialization
s = x.saveobj();
y = spkdec.Convolver.loadobj(s);
for fn = properties(x)', assert(isequal(x.(fn{1}),y.(fn{1}))); end

end



function test_forward_convolution(prm)
% Test the forward convolution
%
% Parameters:
%   do_gpu, nCases, verbose, tol

% Decide on the cases to try
LCDT = randomize_case_dims(prm);
option_cases = struct('use_f32',{false; true}, 'use_gpu',false);
if prm.do_gpu
    option_cases = [option_cases; option_cases];
    [option_cases(3:4).use_gpu] = deal(true);
end
% Verbose header
verbose = prm.verbose;
if verbose
    fprintf('\n');
    fprintf('  %-16s |     Error/eps\n', '  Dimensions');
    fprintf('  %3s %2s %2s %6s | %6s  %6s\n', ...
        'L','D','C','T', 'f64','f32' );
    if prm.do_gpu, fprintf('\b  %6s  %6s\n', 'f64gpu', 'f32gpu'); end
end
% Try each of these cases
errmsg = {};
for ii = 1:prm.nCases
    % Construct the data
    LCD = LCDT(ii,1:3); T = LCDT(ii,4);
    kernels = create_random_kernels(LCD);
    [L,C,D] = size(kernels);
    A = spkdec.Convolver(kernels);
    x = randn(T, D);
    if verbose, fprintf('  %3d %2d %2d %6d |',L,D,C,T); end
    % Compute the reference result
    N_min = 1024; % Otherwise it gets real slow
    y_ref = zeros(T+(L-1),C);
    for d = 1:D
        kern = kernels(:,:,d);
        if (L==1), kern = [kern; zeros(1,C)]; end %#ok<AGROW> % Damnit
        y = fftfilt(kern, [x(:,d); zeros(L-1,1)], N_min);
        y_ref = y_ref + y;
    end
    y_norm = norm(y_ref(:));
    % Try each of our option cases
    for opt = option_cases'
        % Perform the convolution
        x_opt = spkdec.Math.typeconv(x, opt);
        y = A.conv(x_opt);
        assert(isreal(y), 'Real convolution produced complex result');
        % Evaluate the error
        err = y - y_ref;
        err = norm(err(:));
        if opt.use_f32, tol = eps('single'); else, tol = eps('double'); end
        epsilon = tol * y_norm;
        if verbose, fprintf(' %6.1f ',err/epsilon); end
        % Log the error if it exceeds our threshold
        if err > prm.tol*epsilon
            errmsg{end+1} = sprintf(['Excessive error (%.1fx epsilon) in: ' ...
                'L=%d, D=%d, C=%d, T=%d, use_f32=%d, use_gpu=%d\n'], ...
                err/epsilon, L,D,C,T, opt.use_f32, opt.use_gpu); %#ok<AGROW>
        end
    end
    % Go on to the next dimension case
    if verbose, fprintf('\n'); end
end
% Collect the errors
if verbose, fprintf(repmat('_',[1 40])); end
assert(isempty(errmsg), horzcat(errmsg{:}));
end



function test_transpose_convolution(prm)
% Test the transpose convolution
%
% Parameters:
%   do_gpu, nCases, verbose, tol

% Decide on the cases to try
LCDT = randomize_case_dims(prm);
option_cases = struct('use_f32',{false; true}, 'use_gpu',false);
if prm.do_gpu
    option_cases = [option_cases; option_cases];
    [option_cases(3:4).use_gpu] = deal(true);
end
% Verbose header
verbose = prm.verbose;
if verbose
    fprintf('\n');
    fprintf('  %-16s |     Error/eps\n', '  Dimensions');
    fprintf('  %3s %2s %2s %6s | %6s  %6s\n', ...
        'L','D','C','T', 'f64','f32' );
    if prm.do_gpu, fprintf('\b  %6s  %6s\n', 'f64gpu', 'f32gpu'); end
end
% Try each of these cases
errmsg = {};
for ii = 1:prm.nCases
    % Construct the data
    LCD = LCDT(ii,1:3); T = LCDT(ii,4);
    kernels = create_random_kernels(LCD);
    [L,C,D] = size(kernels);
    A = spkdec.Convolver(kernels);
    y = randn(T+(L-1), C);
    if verbose, fprintf('  %3d %2d %2d %6d |',L,D,C,T); end
    % Compute the reference result
    N_min = 1024; % Otherwise it gets real slow
    x_ref = zeros(T,D);
    for d = 1:D
        kern = flipud(conj(kernels(:,:,d)));
        if (L==1), kern = [kern; zeros(1,C)]; end %#ok<AGROW> % Damnit
        x = fftfilt(kern, y, N_min);
        x_ref(:,d) = sum(x(L:end,:), 2);
    end
    x_norm = norm(x_ref(:));
    % Try each of our option cases
    for opt = option_cases'
        % Perform the convolution
        y_opt = spkdec.Math.typeconv(y, opt);
        x = A.convT(y_opt);
        assert(isreal(x), 'Real convolution produced complex result');
        % Evaluate the error
        err = x - x_ref;
        err = norm(err(:));
        if opt.use_f32, tol = eps('single'); else, tol = eps('double'); end
        epsilon = tol * x_norm;
        if verbose, fprintf(' %6.1f ',err/epsilon); end
        % Log the error if it exceeds our threshold
        if err > prm.tol*epsilon
            errmsg{end+1} = sprintf(['Excessive error (%.1fx epsilon) in: ' ...
                'L=%d, D=%d, C=%d, T=%d, use_f32=%d, use_gpu=%d\n'], ...
                err/epsilon, L,D,C,T, opt.use_f32, opt.use_gpu); %#ok<AGROW>
        end
    end
    % Go on to the next dimension case
    if verbose, fprintf('\n'); end
end
% Collect the errors
if verbose, fprintf(repmat('_',[1 40])); end
assert(isempty(errmsg), horzcat(errmsg{:}));
end


function test_convolution_performance(prm)
% Measure the runtime on a problem of a given size
%
% Parameters:
%   do_gpu

% Create the Convolver and example data
kernels = create_random_kernels();
[L,C,D] = size(kernels);
A = spkdec.Convolver(kernels);
T = 2e5;                        % Linear convolution
x = randn(T, D);
y = randn(T+L-1, C);
% Print a header
fprintf('\n');
fprintf('  L=%d,D=%d,C=%d,T=%d\n',L,D,C,T);
fprintf('  %-10s | Runtime (ms)\n', '');
fprintf('  %-10s | %6s  %6s\n', 'Function', 'f64','f32');
if prm.do_gpu, fprintf('\b  %6s  %6s\n', 'f64gpu','f32gpu'); end
% Define the cases
func_handles = {@conv, @convT};
func_inputs = {x, y};
option_cases = struct('use_f32',{false; true}, 'use_gpu',false);
if prm.do_gpu
    option_cases = [option_cases; option_cases];
    [option_cases(3:4).use_gpu] = deal(true);
end
nFuncs = length(func_handles);
% Time each of these
for ii = 1:nFuncs
    func = func_handles{ii};
    fprintf('  %-10s |', func2str(func));
    % Try with each set of options
    for opt = option_cases'
        input = spkdec.Math.typeconv(func_inputs{ii}, opt);
        % Run once to warm up the cache
        func(A, input);
        % Time it for real
        if opt.use_gpu
            runtime = gputimeit(@() func(A,input), 1);
        else
            runtime = timeit(@() func(A,input), 1);
        end
        % Report the result
        fprintf(' %6.1f ', runtime*1e3);
    end
    fprintf('\n');
end
fprintf(repmat('_',[1 40]));
end


function test_fft_scaling_performance(prm)
% Measure the runtime with respect to the FFT size
%
% Parameters:
%   do_gpu

% Create the Convolver and example data
kernels = create_random_kernels();
[L,C,D] = size(kernels);
A = spkdec.Convolver(kernels);
T = 2e5;
% Define the cases
N_arr = pow2(9:16);
option_cases = struct('use_f32',{false; true}, 'use_gpu',false);
option_desc = {'double','float32'};
if prm.do_gpu
    option_cases = [option_cases; option_cases];
    [option_cases(3:4).use_gpu] = deal(true);
    option_desc = [option_desc, {'GPU double','GPU float32'}];
end
nPts = length(N_arr); nOpts = length(option_cases);
% Measure the runtimes over N for a variety of functions
func_names = {'conv()','convT()'};
for funcIdx = 1:2
    figure();
    for optIdx = 1:nOpts
        opt = option_cases(optIdx);
        % Time it for each N
        if opt.use_gpu, time = @gputimeit; else, time = @timeit; end
        lh = loglog(N_arr, NaN(nPts,1), '.-', ...
            'DisplayName',option_desc{optIdx});
        hold on; grid on; title(func_names{funcIdx});
        try
            for ii = 1:nPts
                N = N_arr(ii);
                switch (funcIdx)
                    case 1 % conv()
                        x = spkdec.Math.typeconv(randn(T,D), opt);
                        A.conv(x, 'N_fft',N); % Warmup
                        lh.YData(ii) = time(@() A.conv(x,'N_fft',N),1);
                    case 2 % convT()
                        x = spkdec.Math.typeconv(randn(T+L-1, C), opt);
                        A.convT(x, 'N_fft',N); % Warmup
                        lh.YData(ii) = time(@() A.convT(x,'N_fft',N),1);
                end
                drawnow();
            end
        catch mexc
            % Running out of memory isn't a real failure
            if ~strcmp(mexc.identifier, 'parallel:gpu:array:OOMForOperation')
                rethrow(mexc);
            end
        end
        % Go on to the next computation option
    end
    xlabel('FFT size N'); ylabel('Runtime (s)'); legend('Location','Best');
end
end



% ------------------------------     Helpers     -------------------------------


function kernels = create_random_kernels(LCD)
% Randomly create some problem data
%   kernels = create_random_kernels(LCD)
% Returns:
%   kernel      [L x C x D] spike basis waveforms
% Optional arguments [default]:
%   LCD         [L, C, D] dimensions                [ 125 4 15 ]
if (nargin < 1), LCD = [125 4 15]; end
L = LCD(1); C = LCD(2); D = LCD(3);
% Generate the arrays
kernels = randn(L,C,D);
end


function LCDT = randomize_case_dims(prm)
% Randomly select the dimensions of the test cases to try
%   LCDT = randomize_case_dims(prm)
%
% Returns:
%   LCDT    [nCases x 4] dimensions: [L C D T]
% Required arguments:
%   prm     Struct with fields L_max,C_max,D_max,CDT_max
N = prm.nCases;
L = randi(prm.L_max, [N 1]);
C = randi(prm.C_max, [N 1]);
D = randi(prm.D_max, [N 1]);
T_max = floor(prm.CDT_max ./ (C.*D));
T = ceil(T_max .* rand(N,1));
LCDT = [L C D T];
% Use a deterministic one for the first case
LCDT(1,:) = [125 4 15 20e3];
% And for the next 4 cases, set one of the dimensions to 1 to make sure that we
% can handle edge cases like these
for ii = 1:4
    LCDT(ii+1,ii) = 1;
end
end
