function [basis, spk, resid] = optimize(self, data, varargin)
% Find a spike waveform basis that minimizes our regularized objective function
%   [basis, spk, resid] = optimize(self, data, ...)
%
% Returns:
%   basis       [L x C x D] optimized spike basis waveforms
%   spk         Spikes object (where t is the shift in detected spike time)
%   resid       [L+W-1 x C x N] spike residuals (whitened) after optimization
% Required arguments:
%   data        [L+W-1 x C x N] detected spike waveforms (whitened)
% Optional parameters (key/value pairs) [default]:
%   lambda      Proximal regularizer weight                 [ 0 ]
%   basis_prev  Previous basis (for proximal regularizer)   [ none ]
%   D           Number of basis waveforms overall         [defer to basis_prev]
%   zero_pad    [pre,post] #samples of zero-padding to add  [ 0,0 ]
%   spk_r       [N x 1] sub-sample shift for each spike     [ auto ]
%
% This finds the basis waveforms and spike features to solve:
%     minimize    f(basis,spk) + lambda*||basis-basis_prev||^2
%   subject to    basis is orthonormal
% where the norms are defined in terms of the whitened inner product. The
% objective f() is defined in the BasisOptIncoh class documentation and is a
% combination of the spike reconstruction error and an coherence-penalizing
% regularizer.
%
% If self.dt_search > 0 and/or self.whbasis.R > 1 (unless `spk_r` is given),
% then this process will also search over available full- and/or sub-sample
% shifts in the detected spike times. This reduces the incentive to represent
% such shifts using the spike basis itself.

% --------------------     Problem description     ------------------------

% We can simply reuse all of the optimization framework from the parent class,
% altering individual steps by overloading them.

% In the class documentation, we described the coherence penalty as
%   g(basis) = 1/T * sum_t,n <basis, shift_t*data(:,n)>^2,
% but it would be more accurate to write this as
%   g(basis) = 1/T * sum_t,n ||basis * x(:,n,t)||^2                         (1)
% where
%   x(:,n,t) = argmin_x ||basis*x - shift_t*data(:,n)||^2.                  (2)
% This is just a linear least squares problem. It's slightly complicated by our
% whitened inner product, defined by
%   <x,y> = x' * W * y,
% where W is a symmetric, positive definite matrix. Solving the least squares
% problem in (2) gives us
%   x(:,n,t) = (basis'*W*basis) \ (basis' *W* shift_t*data(:,n)).           (3)
% But since our basis is orthonormal, basis'*W*basis == I and so this reduces to
%   x(:,n,t) = basis' * W * shift_t*data(:,n)
%            = <basis, shift_t*data(:,n)>.                                  (4)
% This is a slight abuse of notation, since it is a vector-valued dot product
% between a matrix (basis) and a vector (shift_t*data_n), but I think it's more
% readable this way. Anyways, the orthonormality also gives us
%   ||basis*x||^2 = x' * basis' * W * basis * x = x'*x = ||x||^2.
% Remember that norms on spike waveforms like ||basis*x|| use the whitened inner
% product, while all other norms, like ||x||, are the standard Euclidean norm.
% Substituting this and eq (4) into eq (1) finally gives us
%   g(basis) = 1/T * sum_t,n ||basis' * W * shift_t*data(:,n)||^2           (5)
%            = 1/T * sum_t,n <basis, shift_t*data(:,n)>^2.
% We have again abused our notation to take the norm of a vector-valued dot
% product, but anyways, that's what's presented in the class documentation.

% However, our use of sub-sample shifts complicates this somewhat. First, let's
% shorten our notation. First, let's note that we have a matrix wh_01 such that
% wh_01*wh_01' == W, and we can incorporate that into our definitions in order
% to avoid carrying around all of these W's. Let:
%   A = wh_01 * basis
%   St = wh_01 * shift_t   <-- assuming that shift_t is in unwhitened space
%   Sr = shift matrix (in Q1 coordinates) for a sub-sample shift of r.
% We can then define our coherence penalty as:
%   g(A) = 1/RT * sum_t,r,n ||A * x(:,n,t,r)||^2                            (6)
% where
%   x(:,n,t,r) = argmin_x ||Sr*A - St*data(:,n)||^2
%              = (A'*Sr'*Sr*A) \ (A'*Sr' * St*data(:,n))                    (7)
% Unfortunately, the introduction of Sr means that this first term is no longer
% equivalent the identity matrix. As a result, we end up with the far more
% cumbersome definition
%   g(A) = 1/RT * sum_t,r,n ||A * Mr \ A'*Sr'*St*data(:,n)||^2,             (8)
% where
%   Mr = A'*Sr' * Sr*A.                                                     (9)
% There are a few steps we can take to express (8) more concisely. First, we can
% replace the sum over n with the Frobenius norm:
%   g(A) = 1/RT * sum_t,r ||A * Mr \ A'*Sr'*St*data||^2.                    (10)
% Expanding this Frobenius norm using the trace:
%   ||A*inv(Mr)*A'*Sr'*St*data||^2
%    = Tr(data'*St'*Sr*A*inv(Mr)*A'*A*inv(Mr)*A'*Sr'*St*data)
% We can then rely on two properties of the trace (linearity and invariance to
% cyclic permutations) to simplify the sum over t:
%   1/T * sum_t ||A * Mr \ A'*Sr'*St*data||^2
%    = 1/T * sum_t Tr(Sr*A*inv(Mr)*A'*A*inv(Mr)*Sr'*St*data*data'*St')
%    = Tr(Sr*A * inv(Mr)*A'*A*inv(Mr) * A'*Sr' * B),
% where
%   B = 1/T * sum_t St*data*data'*St'                                       (11)
% is the mean (over the full-sample shifts t) of the data covariance, and it is
% a constant (it does not depend on the basis A). Substituting this back into
% (10), we get
%   g(A) = 1/R * sum_r Tr(Sr*A * inv(Mr)*A'*A*inv(Mr) * A'*Sr' * B).        (12)
% The trace is not very computationally efficient, so it may be convenient to
% introduce the lower Cholesky decomposition of B:
%   Lb*Lb' = B = 1/T * sum_t St*data*data'*St'                              (13)
% so that we can express (12) as
%   g(A) = 1/R * sum_r Tr(Lb'*Sr*A*inv(Mr)*A' * A*inv(Mr)*A'*Sr'*Lb)
%        = 1/R * sum_r ||A * inv(Mr) * A'*Sr' * Lb||^2                      (14)

% So that's the coherence-penalizing regularizer g(A). It's defined in (6) and
% simplified until we arrive at (14). We also need its gradient. I'll refer you
% to the Matrix Cookbook section on trace derivatives, but applying these rules
% to the summand of (12), and dropping the subscripts for the sake of brevity,
%   grad_A Tr(S*A*inv(M)*A'*A*inv(M)*A'*S'*B)
%    = 2*(  S'*B*S*A*inv(M)*A'*A*inv(M) + A*inv(M)*A'*S'*B*S*A*inv(M) 
%         - S'*S*A*inv(M)*A'*A*inv(M)*A'*S'*B*S*A*inv(M)
%         - S'*S*A*inv(M)*A'*S'*B*S*A*inv(M)*A'*A*inv(M)   )
%    = 2*(I - S'*S*A*inv(M)*A') * (S'*B*S*A*inv(M)*A' + [...]') * A*inv(M). (15)
% where [...]' refers to the transpose of the preceding term.


% Call the superclass method
args_out = cell(nargout,1);
[args_out{:}] = optimize@spkdec.BasisOptimizer(self, data, varargin{:});

% Assign output arguments
if nargout >= 1, basis = args_out{1}; end
if nargout >= 2, spk   = args_out{2}; end
if nargout >= 3, resid = args_out{3}; end

end
