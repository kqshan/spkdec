function err = eval_error(self, A, X)
% Evaluate the spike reconstruction error given a basis and spikes
%   err = eval_error(self, A, X)
%
% Returns:
%   err   [L*C x N] reconstruction error map_21*A*X - Y
% Required arguments:
%   A     Whitened spike basis (see get_shifted_basis for format)
%   X     Output struct from optimize_spk()

% Dimensions
N = length(X.s);
R = self.R;

% Start by setting err = -Y (and applying the selected full-sample shifts)
spk_ns = (1:N)' + N*(X.s-1);
err = -self.Y(:, spk_ns);

% Group the spikes by sub-sample shift
r_spkidx = accumarray(X.r, (1:N)', [R 1], @(x) {x});

% Perform the reconstruction
for r = 1:R
    % Build A2 = map_21r * A
    A2 = self.get_shifted_basis(A, r);
    
    % Perform the reconstruction for the selected spikes
    spkidx = r_spkidx{r};
    err(:,spkidx) = err(:,spkidx) + A2 * X.X(:,spkidx);
end

end
