function consts_init(self, A, b, beta)
% Initialize the temporary cache of problem-specific constants
%   init_problem_consts(self, A, b, beta)
%
% Required arguments:
%   A       Convolution kernels for this problem (SpikeBasis object)
%   b       [T+V x C] data to approximate
%   beta    Regularizer cost per spike

t_start = tic();

% Check that the dimensions match
[Tb, C] = size(b);
assert(Tb > A.V, self.errid_dim, ...
    'The given data is too short for the chosen spike basis');
assert(C == A.C, self.errid_dim, ...
    'The data and spike basis must have the same number of channels');

% Compute the selection refractory period based on the coh_thresh property
gram = A.toGram();
lag_norms = gram.lagNorms();
norm_thresh = self.coh_thresh * min(svd(gram.getGram(0)));
select_dt = find(lag_norms > norm_thresh, 1, 'last');

% Populate the cache
self.A    = A;
self.b    = b;
self.At_b = A.convT(b);
self.beta = beta;
self.select_dt = select_dt;
self.t_start   = t_start;

end
