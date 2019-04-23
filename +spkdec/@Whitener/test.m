function test(varargin)
% Run unit tests on this class
%   Whitener.test( ... )
%
% Optional parameters (key/value pairs) [default]:
%   do_gpu      Run GPU-based tests             [ auto ]
%   nCases      #cases for varying dimensions   [ 10 ]
%   tol         Relative error tolerance        [ 10 ]
%   verbose     Display more output             [ false ]
%   rethrow     Rethrow error immediately       [ false ]

% Optional params
ip = inputParser();
ip.addParameter('do_gpu', [], @(x) isempty(x) || isscalar(x));
ip.addParameter('W_max', 400, @isscalar);
ip.addParameter('C_max',  90, @isscalar);
ip.addParameter('WCT_max', 1e6, @isscalar);
ip.addParameter('nCases', 10, @(x) isscalar(x) && x >= 6);
ip.addParameter('tol', 20, @isscalar);
ip.addParameter('verbose', false, @isscalar);
ip.addParameter('rethrow', false, @isscalar);
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
    % Run the test
    fprintf('%s%s', test_str, repmat('.',1,max(3,40-length(test_str))));
        test_func(prm);
    try
        fprintf('OK\n');
    catch mexc
        if prm.rethrow, rethrow(mexc); end
        fprintf('FAIL\n');
        errmsgs{end+1} = sprintf('%s: %s',test_str,mexc.message); %#ok<AGROW>
    end
end

% Throw an error
assert(isempty(errmsgs), 'spkdec:Whitener:TestFail', ...
    sprintf('%s\n', errmsgs{:}));

end


% -----------------------------     Unit tests     -----------------------------


function test_object_management(~)
% Run tests on basic object management (construction, copy, serialization)

% Default constructor
x = spkdec.Whitener();
assert(x.C==1 && x.W==1 && x.delay == 0);
% Constructor with arguments
W = randi(200); C = randi(60);
wh_filt = randn(W,C); wh_ch = randn(C,C); delay = randi(W);
x = spkdec.Whitener('wh_filt',wh_filt, 'wh_ch',wh_ch, 'delay',delay);
assert(isequal(x.wh_filt,wh_filt) && isequal(x.wh_ch,wh_ch));
assert(x.W==W && x.C==C);
% Copy
y = x.copy();
for fn = properties(x)', assert(isequal(x.(fn{1}),y.(fn{1}))); end
% Serialization
s = x.saveobj();
y = spkdec.Whitener.loadobj(s);
for fn = properties(x)', assert(isequal(x.(fn{1}),y.(fn{1}))); end

end


function test_makeWhCh(prm)
% Run unit tests on the makeWhCh() static method
C = 7;
verbose = prm.verbose;
% Generate a covariance matrix
ch_cov = cov(randn(50*C, C));
% Try the different methods for whitening it
if verbose, fprintf('\n%8s  rel_err\n','method'); end
for method_cell = {'eig','chol','sqrtm'}
    % Call the function under test
    method = method_cell{1};
    wh_ch = spkdec.Whitener.makeWhCh(ch_cov, 'method',method);
    % Check if this has whitened the given covariance
    wh_cov = wh_ch * ch_cov * wh_ch';
    err = wh_cov - eye(C);
    % Compare this to the allowed tolerance
    rel_err = abs(err) / eps();
    rel_err = max(rel_err(:));
    if verbose, fprintf('%8s  %7.1f\n',method,rel_err); end
    assert(rel_err < prm.tol, 'Excessive whitener error');
end
if verbose, fprintf(repmat('_',[1 40])); end
end


function test_whitening(prm)
% Test the whitening
%
% Parameters:
%   do_gpu, W_max, C_max, WCT_max, nCases, verbose, tol

% Decide on the cases to try
WCT = randomize_case_dims(prm);
option_cases = struct('use_f32',{false; true}, 'use_gpu',false);
if prm.do_gpu
    option_cases = [option_cases; option_cases];
    [option_cases(3:4).use_gpu] = deal(true);
end
% Verbose header
verbose = prm.verbose;
if verbose
    fprintf('\n');
    fprintf('  %-12s |     Error/eps\n', '  Dimensions');
    fprintf('  %3s %2s %6s | %6s  %6s\n', ...
        'W','C','T', 'f64','f32' );
    if prm.do_gpu, fprintf('\b  %6s  %6s\n', 'f64gpu', 'f32gpu'); end
end
% Try each of these cases
errmsg = {};
for ii = 1:prm.nCases
    % Construct the data
    W = WCT(ii,1); C = WCT(ii,2); T = WCT(ii,3);
    whitener = create_random_whitener(W,C);
    x_ref = randn(T, C);
    if verbose, fprintf('  %3d %2d %6d |',W,C,T); end
    % Compute the reference result
    N_min = 1024; % Otherwise it gets real slow
    y_ref = zeros(T+W-1, C);
    for c = 1:C
        y_ref(:,c) = fftfilt(whitener.wh_filt(:,c), ...
            [x_ref(:,c); zeros(W-1,1)], N_min);
    end
    y_ref = (whitener.wh_ch * y_ref.').';
    y_norm = norm(y_ref(:));
    % Try each of our option cases
    for opt = option_cases'
        % Perform the convolution
        x = spkdec.Math.typeconv(x_ref, opt);
        y = whitener.whiten(x, 'bounds','keep');
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
                'W=%d, C=%d, T=%d, use_f32=%d, use_gpu=%d\n'], ...
                err/epsilon, W,C,T, opt.use_f32, opt.use_gpu); %#ok<AGROW>
        end
        % Evaluate the other truncation options
        y2 = whitener.whiten(x, 'bounds','trunc');
        assert(isequal(y2, y(W:end-W+1,:)));
        y2 = whitener.whiten(x, 'bounds','center');
        assert(isequal(y2, y(whitener.delay+(1:T),:)));
    end
    % Go on to the next dimension case
    if verbose, fprintf('\n'); end
end
% Collect the errors
if verbose, fprintf(repmat('_',[1 40])); end
assert(isempty(errmsg), horzcat(errmsg{:}));
end


% ------------------------------     Helpers     -------------------------------


function wh = create_random_whitener(W, C)
% Randomly create some problem data
%   wh = create_random_whitener(CW)
% Returns:
%   wh      spkdec.Whitener object
% Required arguments:
%   W,C     Dimensions
wh = spkdec.Whitener('wh_filt',randn(W,C), ...
    'wh_ch',randn(C,C), 'delay',randi(W)-1);
end


function WCT = randomize_case_dims(prm)
% Randomly select the dimensions of the test cases to try
%   WCT = randomize_case_dims(prm)
%
% Returns:
%   WCT     [nCases x 3] dimensions: [W C T]
% Required arguments:
%   prm     Struct with fields W_max,C_max,WCT_max
N = prm.nCases;
W = randi(prm.W_max, [N 1]);
C = randi(prm.C_max, [N 1]);
T_max = floor(prm.WCT_max ./ (W.*C));
T = ceil(T_max .* rand(N,1));
WCT = [W C T];
% Use a deterministic one for the first case
WCT(1,:) = [250 4 20e3];
% And for the next cases, set one of the dimensions to 1 to make sure that we
% can handle edge cases like these
for ii = 1:size(WCT,2)
    WCT(ii+1,ii) = 1;
end
end
