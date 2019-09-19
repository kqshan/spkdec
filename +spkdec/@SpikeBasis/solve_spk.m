function X = solve_spk(self, waveforms, spk_r)
% Solve for the optimal spike features given the extracted spike waveforms
%   spk_X = solve_spk(self, waveforms, spk_r)
%
% Returns;
%   spk_X       [D x N] optimal spike features
% Required arguments:
%   waveforms   [L+W-1 x C x N] extracted spike waveforms
%   spk_r       [N x 1] spike sub-sample shift index (1..R)

% Basically, we define
%   Y = reshape(waveforms, [Lw*C, N])
%   A = reshape(self.toKern(), [Lw*C, D])
% and then we find
%   X = A \ Y

% Note 1: This is slightly complicated by the fact that we have multiple
% sub-sample shifts to choose from.

% Note 2: We actually have the Cholesky decomposition of A'*A cached, so we
% could use that to help speed up the X = A\Y operation, but meh

% Get A
A = self.toKern();
[Lw, C, D, R] = size(A);
A = reshape(A, [Lw*C, D, R]);

% Reshape the data
[Lw_, C_, N] = size(waveforms);
assert(Lw_==Lw && C_==C, self.errid_dim, 'waveforms must be [L+W-1 x C x N]');
Y = reshape(waveforms, [Lw*C, N]);

% Solve the linear inverse problem for each sub-sample shift
X = zeros(D, N, 'like',Y);
R = self.R;
r_idx = accumarray(spk_r(:), (1:N)', [R 1], @(x) {x});
for r = 1:R
    idx = r_idx{r};
    X(:,idx) = A(:,:,r) \ Y(:,idx);
end

end
