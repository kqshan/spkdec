function err = eval_error(self, A, X)
% Evaluate the spike reconstruction error given a basis and spikes
%   err = eval_error(self, A, X)
%
% Returns:
%   err   [L*C x N] reconstruction error map_21*A*X - Y
% Required arguments:
%   A     [L x K x C] whitened spike basis waveforms in Q2 basis
%   X     Output struct from optimize_spk()

% Dimensions and local variables
[L, K, C] = size(A);
N = length(X.s);
R = self.R;
whbasis = self.whbasis;

% Start by building a composite Y from the sub-sample shifts
spk_ns = (1:N)' + N*(X.s-1);
err = -self.Y(:, spk_ns);

% Group the spikes by sub-sample shift
r_spkidx = accumarray(X.r, (1:N)', [R 1], @(x) {x});

% Perform the reconstruction
for r = 1:R
    % Build A2 = map_21r * A (slightly complicated by the storage format of A)
    A2 = zeros(L*C, K, C);
    for c = 1:C
        A2(:,:,c) = whbasis.map_21r(:,:,c,r) * A(:,:,c);
    end
    A2 = reshape(A2, [L*C, K*C]);
    
    % Perform the reconstruction for the selected spikes
    spkidx = r_spkidx{r};
    err(:,spkidx) = err(:,spkidx) + A2 * X.X(:,spkidx);
end

end
