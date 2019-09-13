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

% Call the superclass method
A = start_optimization@spkdec.BasisOptimizer(self, data, prm);

% Compute coh_YYt
% Get some dimensions and local variables
[LC,~,S] = size(self.Y);
L = self.L; C = self.C; assert(LC==L*C);
R = self.R;
Q1 = self.whbasis.Q1;
Lw = size(Q1,1);
Sr = self.whbasis.shift1r;
% Compute Y*Y' for the one that corresponds to dt==0
s_idx = (S-1)/2 + 1;
assert(mod(s_idx,1)==0);
Y = self.Y(:,:,s_idx);
YYt = gather(double(Y*Y'));
% Sum this over +/- L with of shifts
coh_YYt = zeros(LC,LC);
for dt = 0:L
    % Construct the shift operator in Q1 coordinates
    shift = reshape(Q1(dt+1:end,:,:), [(Lw-dt)*C, LC])' ...
        * reshape(Q1(1:end-dt,:,:), [(Lw-dt)*C, LC]);
    % Sum over sub-sample shifts as well
    for r = 1:R
        S = Sr(:,:,r)' * shift;
        coh_YYt = coh_YYt + S*YYt*S';
        % And include negative shifts too
        if (dt > 0)
            S = Sr(:,:,r)' * shift';
            coh_YYt = coh_YYt + S*YYt*S';
        end
        % Go on to the next sub-sample shift
    end
    % Go on to the next full-sample shift
end
T = R * (2*L+1);
coh_YYt = coh_YYt / T;
% Store this result
self.coh_YYt = coh_YYt;

% Compute coh_L
self.coh_L = chol(coh_YYt, 'lower');

end
