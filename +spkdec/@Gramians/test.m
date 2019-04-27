function test(varargin)
% Run unit tests on this class
%   Gramians.test( ... )
%
% Optional parameters (key/value pairs) [default]:
%   nCases      #cases for varying dimensions   [ 10 ]
%   tol         Relative error tolerance        [ 10 ]
%   verbose     Display more output             [ false ]

% Optional params
ip = inputParser();
ip.addParameter('L_max', 500, @isscalar);
ip.addParameter('C_max',  64, @isscalar);
ip.addParameter('D_max', 256, @isscalar);
ip.addParameter('R_max',   9, @isscalar);
ip.addParameter('D2R_max', 512*1024, @isscalar);
ip.addParameter('nCases', 10, @(x) isscalar(x) && x >= 6);
ip.addParameter('tol', 10, @isscalar);
ip.addParameter('verbose', false, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Run the individual unit tests
errmsgs = {};
for test_func_cell = localfunctions()'
    test_func = test_func_cell{1};
    % Ignore any local functions that don't start with "test"
    test_str = func2str(test_func);
    if ~startsWith(test_str,'test'), continue; end
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
assert(isempty(errmsgs), 'spkdec:Gramians:TestFail', ...
    sprintf('%s\n', errmsgs{:}));

end


% -----------------------------     Unit tests     -----------------------------


function test_object_management(~)
% Run tests on basic object management (construction, copy, serialization)

% Constructor
kernels = create_random_kernels();
[L, C, D, R] = size(kernels);
x = spkdec.Gramians(kernels);
assert(isequal(x.kernels,kernels));
assert(x.L==L && x.C==C && x.D==D && x.R==R);
% Copy
y = x.copy();
for fn = properties(x)', assert(isequal(x.(fn{1}),y.(fn{1}))); end
% Serialization
s = x.saveobj();
y = spkdec.Gramians.loadobj(s);
for fn = properties(x)', assert(isequal(x.(fn{1}),y.(fn{1}))); end

end


function test_gram_matrix(prm)
% Run tests on the getGram() member function

% Decide on the cases to try
LCDR = randomize_case_dims(prm);
% Verbose header
verbose = prm.verbose;
if verbose
    fprintf('\n');
    fprintf('  %3s %3s %3s %3s | lag,r1,r2:error/eps\n', 'L','C','D','R');
end
% Try each of these cases
errmsg = {};
for ii = 1:prm.nCases
    % Construct the data
    kernels = create_random_kernels(LCDR(ii,:));
    [L,C,D,R] = size(kernels);
    gramians = spkdec.Gramians(kernels);
    if verbose, fprintf('  %3d %3d %3d %3d |', L, C, D, R); end
    % Try some different combinations of lag, r1, r2
    LRR = L * R * R;
    N = min(4, LRR);
    for lrr = [0, randperm(LRR-1,N-1)]
        lag = mod(lrr,L);             % 0..L-1
        r1 = mod(floor(lrr/L),R) + 1; % 1..R
        r2 = floor(lrr/L/R) + 1;      % 1..R
        % Also randomize the sign of the lag
        if rand()<0.5, lag = -lag; end
        % Compute the reference result
        x1 = zeros(3*L,C,D);
        x2 = zeros(3*L,C,D);
        x1(L  +  (1:L),:,:) = kernels(:,:,:,r1);
        x2(L+lag+(1:L),:,:) = kernels(:,:,:,r2);
        x1 = reshape(x1,[3*L*C,D]);
        x2 = reshape(x2,[3*L*C,D]);
        G_ref = x1' * x2;
        G_norm = norm(x1,'fro') * norm(x2,'fro');
        
        % Compute the result using the object
        G = gramians.getGram(lag, r1, r2);
        % Evaluate the error
        err = norm(G-G_ref,'fro') / G_norm;
        rel_err = err / eps();
        if verbose, fprintf('%4d,%d,%d:%.1f ',lag,r1,r2,rel_err); end
        % Log the error if it exceeds our threshold
        if rel_err > prm.tol
            errmsg{end+1} = sprintf(['Excessive error (%.1fx epsilon) in: ' ...
                'L=%d, C=%d, D=%d, R=%d, lag=%d, r1=%d, r2=%d\n'], ...
                rel_err, L,C,D,R, lag,r1,r2); %#ok<AGROW>
        end
    end
    % Go on to the next dimension
    if verbose, fprintf('\n'); end
end
% Collect the errors
if verbose, fprintf(repmat('_',[1 40])); end
assert(isempty(errmsg), horzcat(errmsg{:}));
end


function test_gram_sequence(prm)
% Run tests on the getGramSeq() member function

% Decide on the cases to try
LCDR = randomize_case_dims(prm);
% Verbose header
verbose = prm.verbose;
if verbose
    fprintf('\n');
    fprintf('  %3s %3s %3s %3s | error/eps\n', 'L','C','D','R');
end
% Try each of these cases
errmsg = {};
for ii = 1:prm.nCases
    % Construct the data
    kernels = create_random_kernels(LCDR(ii,:));
    [L,C,D,R] = size(kernels);
    gramians = spkdec.Gramians(kernels);
    if verbose, fprintf('  %3d %3d %3d %3d | ', L, C, D, R); end
    % Select the spike times and kernel families
    N = 10; % Number of spikes
    dt = randi(L, [N 1]);
    spk_t = cumsum(dt) - dt(1);
    spk_r = randi(R, [N 1]);
    % Compute the reference result
    T = max(spk_t);
    x = zeros(L+T, C, D, N);
    for n = 1:N
        x(spk_t(n) + (1:L), :, :, n) = kernels(:,:,:,spk_r(n));
    end
    x = reshape(x, [(L+T)*C, D*N]);
    G_ref = x' * x;
    
    % Compute the result using the Gramians class
    G_bands = gramians.getGramSeq(spk_t, spk_r);
    G = spkdec.Math.symband_to_sparse(G_bands);
    % Evaluate the error
    err = norm(G-G_ref,'fro') / norm(G_ref,'fro');
    rel_err = err/eps();
    if verbose, fprintf('%6.1f\n',rel_err); end
    % Log the error if it exceeds our threshold
    if rel_err > prm.tol
        errmsg{end+1} = sprintf(['Excessive error (%.1fx epsilon) in: ' ...
            'L=%d, C=%d, D=%d, R=%d'], rel_err, L,C,D,R); %#ok<AGROW>
    end
end
% Collect the errors
if verbose, fprintf(repmat('_',[1 40])); end
assert(isempty(errmsg), horzcat(errmsg{:}));
end


function test_cache(~)
% Test the Gram matrix caching

% Construct the object
kernels = create_random_kernels();
[L,C,D,R] = size(kernels); %#ok<ASGLU>
gramians = spkdec.Gramians(kernels);
% Select some lag,r1,r2 to query
N = 100; assert(N <= L*R*R);
lrr = randperm(L*R*R, N) - 1;
lag = mod(lrr,L);             % 0..L-1
r1 = mod(floor(lrr/L),R) + 1; % 1..R
r2 = floor(lrr/L/R) + 1;      % 1..R
% Reduce the cache size so that it can only hold half as many as this
cache_size = floor(N/2);
gramians.gmc_revidx = gramians.gmc_revidx(1:cache_size);
gramians.gmc_data = gramians.gmc_data(:,:,1:cache_size);
gramians.gmc_size = cache_size;
% Call it once to get the reference result
G_ref = gramians.getGram(lag, r1, r2);
% Call it again, which should result in some of these being served by the cache
G = gramians.getGram(lag, r1, r2);
% Check that these are exactly equal
assert(isequal(G, G_ref), 'Gram matrix cache did not return exact results');
end



% ------------------------------     Helpers     -------------------------------


function kernels = create_random_kernels(LCDR)
% Randomly create some problem data
%   kernels = create_random_kernels(LCDR)
% Returns:
%   kernel      [L x C x D x R] whitened kernels
% Optional arguments [default]:
%   LCDR        [L, C, D, R] dimensions             [ 250 4 12 3 ]
if (nargin < 1), LCDR = [250 4 12 3]; end
L = LCDR(1); C = LCDR(2); D = LCDR(3); R = LCDR(4);
% Generate the kernels
kernels = randn(L,C,D,R);
end


function LCDR = randomize_case_dims(prm)
% Randomly select the dimensions of the test cases to try
%   LCDR = randomize_case_dims(prm)
%
% Returns:
%   LKCT    [nCases x 4] dimensions: [L C D R]
N = prm.nCases;
L = ceil((sqrt(prm.L_max) .* rand(N,1)).^2);
C = ceil((sqrt(prm.C_max) .* rand(N,1)).^2);
R = randi(prm.R_max, [N 1]);
D_max = floor(sqrt(prm.D2R_max./R));
D_max = min(D_max, prm.D_max);
D = ceil((sqrt(D_max) .* rand(N,1)).^2);
LCDR = [L C D R];
% Use a deterministic one for the first case
LCDR(1,:) = [250 4 12 3];
% And for the next 4 cases, set one of the dimensions to 1 to make sure that we
% can handle edge cases like these
for ii = 1:4
    LCDR(ii+1,ii) = 1;
end
end
