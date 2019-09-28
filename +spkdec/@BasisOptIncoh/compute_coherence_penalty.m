function g = compute_coherence_penalty(self, A, B)
% Compute the coherence penalty evaluated on the given spike basis
%   g = compute_coherence_penalty(self, A)
%
% Returns:
%   g       Coherence penalty g(A)
% Required arguments:
%   A       [L*C x D] whitened spike basis waveforms in Q1 coordinates
%
% This can also be called with 3 arguments, in which case this computes
%   g_diff = coherence_penalty(self, A, B)
% Returns:
%   g_diff  Coherence penalty difference g(A) - g(B)
% Required arguments:
%   A,B     [L*C x D] whitened spike basis waveforms in Q1 coordinates
% This form is numerically more accurate than calling it on A and B separately.
if nargin < 3, B = []; end

% The g(A) we're evaluating is defined as (see the comments in optimize.m)
%   g(A) = 1/R * sum_r ||A * inv(A'*Sr'*Sr*A) * A'*Sr' * Lb||^2             (1)
% where
%   Sr = shift matrix (in Q1 coordinates) for a sub-sample shift of r
%   Lb*Lb' = 1/T * sum_t Q1'*shift_t*data*data'*shift_t'*Q1 = coh_YYt
%
% For a skinny matrix X, the backslash operator computes
%   X \ Y = inv(X'*X) * X'*Y
% so the expression in (1) can be simplified to
%   g(A) = 1/R * sum_r ||A * (Sr*A) \ Lb||^2

% If we further assume that A is orthonormal, then we can remove the A from the
% norm. But computing the QR decomposition of A is computationally trivial so
% we'll just do that instead, and maybe that'll save us some trouble in the
% future.
[~, A_norms] = qr(A,0);
if ~isempty(B), [~,B_norms] = qr(B,0); end

% Collect some dimensions and local variables
Sr = self.whbasis.shift1r;
Lb = self.coh_L;
R = self.R;

% Compute g(A)  <--  Or g(A)-g(B) if B is given
g = 0;
for r = 1:R
    % Compute the summand
    Ar = Sr(:,:,r) * A;
    gr = (A_norms * (Ar\Lb)).^2;
    % Subtract the summand for B if desired
    if ~isempty(B)
        Br = Sr(:,:,r) * B;
        gr = gr - (B_norms * (Br\Lb)).^2;
    end
    % Sum and update the accumulator
    g = g + sum(gr,'all')/R;
end

end
