function A = start_optimization(self, data, prm)
% Initialize the optimization routine based on the given data
%   A = start_optimization(self, data, prm)
%
% Returns:
%   A           [L*C x D] whitened spike basis in Q1 coordinates
% Required arguments:
%   data        [L+W-1 x C x N] detected spike waveforms (whitened)
%   prm         Struct of additional parameters, with fields
%     lambda      Proximal regularizer weight
%     basis_prev  Previous basis (for proximal regularizer)
%     D           Number of basis waveforms overall (if basis_prev not given)
%     zero_pad    [pre,post] #samples of zero-padding to add
%     spk_r       [N x 1] sub-sample shift for each spike (may be empty)
%
% This also defines the following protected properties:
%   Y           [L*C x N x S] spike data in Q1 coordinates, with the S
%               dimension corresponding to shifts of (-dt_search:dt_search)
%   A0          [L*C x D] whitened previous basis in Q1 coordinates
%   lambda      Weight applied to the proximal regularizer ||A-A0||
%   spk_r       User-specified sub-sample shift for each spike
%   coh_YYt     [L*C x L*C] mean of Y*Y' with different amount of shift applied
%   coh_L       Cholesky decomposition of coh_YYt: coh_L*coh_L' == coh_YYt
%   grad_g      Gradient of the coherence penalty g(A)

% Call the superclass method
A = start_optimization@spkdec.BasisOptimizer(self, data, prm);

% Compute coh_YYt
% Get some dimensions and local variables
[Lw,C,N] = size(data);
L = self.L;
Q1 = self.whbasis.Q1;
% Compute data_cov = data'*data as a [Lw x C x Lw x C] array
data_cov = reshape(data, [Lw*C, N]);
data_cov = gather(data_cov * data_cov');    % [Lw*C x Lw*C]
data_cov = reshape(data_cov,[Lw C Lw C]);
% Sum Q1'*data_cov*Q1 over +/- L worth of shifts
coh_YYt = zeros(L*C, L*C);
for dt = -L:L
    % Get the shifted versions of Q1 and the data
    dt_pos = max(dt,0);
    dt_neg = min(dt,0);
    dcov_s = data_cov(1+dt_pos:Lw+dt_neg, :, 1+dt_pos:Lw+dt_neg, :);
    Q1_s = Q1(1-dt_neg:Lw-dt_pos, :, :);
    % Compute the contribution
    Lw_s = Lw - abs(dt);
    dcov_s = reshape(dcov_s, [Lw_s*C, Lw_s*C]);
    Q1_s = reshape(Q1_s, [Lw_s*C, L*C]);
    YYt_s = Q1_s' * dcov_s * Q1_s;
    % Add this to our accumulator
    coh_YYt = coh_YYt + double(YYt_s);
end
T = 2*L + 1;
coh_YYt = coh_YYt / T;
% Store this result
self.coh_YYt = coh_YYt;

% Compute coh_L
assert(T*N >= L*C);
self.coh_L = chol(coh_YYt, 'lower');

% Place an empty filler into grad_g
self.grad_g = [];

end
