function spk = optimize_spk(self, A)
% Optimize the spikes with the given basis
%   X = optimize_spk(self, A)
%
% Returns:
%   X       Struct with fields:
%     X       [D x N] spikes in feature space
%     r       [N x 1] sub-sample shift index (1..R)
%     s       [N x 1] full-sample shift index (1..S)
%     X_cov   [D x D x R] matrix such that Xcov*X_cov' == X*X' for each r
%     ErrXt   [L*C x D x R] Error*X' for each sub-sample shift index (1..R)
% Required arguments:
%   A       Whitened spike basis (see get_shifted_basis for format)
%
% This solves the problem
%   minimize (X)     ||mat_21*A*X - Y||
% But we also need to search over the R sub-sample and S full-sample shifts:
%   minimize (r,s,X) ||mat_21r(:,:,r)*A*X - Y(:,:,s)||

% 0. Preparatory steps ---------------------------------------------------------

% Get some dimensions
[LC, N, S] = size(self.Y);
[~, D] = size(A);
R = self.R;

% Perform QR decompositions of A2r = mat_21r * A for each sub-sample shift
A2r_Q = zeros(LC, D, R);
A2r_R = zeros(D, D, R);
for r = 1:R
    % Construct the [L*C x D] basis (in Q1 coords) for this sub-sample shift
    A2r = self.get_shifted_basis(A, r);
    % Perform the QR decomposition
    [A2r_Q(:,:,r), A2r_R(:,:,r)] = qr(A2r, 0);
end

% 1. Find the best shift index for each spike ----------------------------------

% Now that we've performed a QR decomposition, we can solve
%   X = (A2'*A2) \ A2'*Y 
%     = R \ Q'*Y
% And to evaluate which is the best shift index, let us compute
%   delta = ||y||^2 - ||y-A2*x||^2 = ||A2*x||^2 = ||Q'*y||^2
delta = zeros(N, R, S, 'like',self.Y);
if (S==1) && (R==1 || ~isempty(self.spk_r))
    % Our choice of shift index is fully constrained, no need to compute delta
    % Leaving it as a [N x 1] vector of zeros still yields the correct result
else
    % Compute delta for each (r,s)
    for s = 1:S
        for r = 1:R
            delta(:,r,s) = sum((A2r_Q(:,:,r)' * self.Y(:,:,s)).^2, 1)';
        end
    end
end

% Find the optimal (s,r) for each spike
[~,spk_rs] = max(reshape(delta,[N, R*S]), [], 2);
spk_rs = gather(spk_rs);
spk_s = ceil(spk_rs/R);
spk_r = spk_rs - R*(spk_s-1);

% Overwrite spk_r if desired
if ~isempty(self.spk_r)
    spk_r = self.spk_r;
end

% 2. Solve for X and compute the additional quantities -------------------------

% Build a composite Y using the selected full-sample shifts
spk_ns = (1:N)' + N*(spk_s-1);
Y = self.Y(:, spk_ns);

% Group the spikes by sub-sample shift
r_spkidx = accumarray(spk_r, (1:N)', [R 1], @(x) {x});

% Process each sub-sample shift
X = zeros(D, N, 'like',Y);
X_cov = zeros(D, D, R);
ErrXt = zeros(LC, D, R);
for r = 1:R
    % Solve for X = R \ Q' * Y (and let Z = Q' * Y)
    spkidx = r_spkidx{r};
    Zr = A2r_Q(:,:,r)' * Y(:,spkidx);
    Xr = A2r_R(:,:,r) \ Zr;
    X(:,spkidx) = Xr;
    % Find some X_cov such that X_cov*X_cov' == X*X'
    Nr = length(spkidx);
    if Nr <= D
        X_cov(:,:,r) = [gather(double(Xr)), zeros(D,D-Nr)];
    else
        X_cov(:,:,r) = chol(gather(double(Xr*Xr')), 'lower');
    end
    % Compute the reconstruction error
    %   Err = A2*X - Y = Q*R*(R\Z) - Y = Q*Z - Y
    Err = A2r_Q(:,:,r) * Zr - Y(:,spkidx);
    ErrXt(:,:,r) = gather(double(Err * Xr'));
end

% 3. Package this for output ---------------------------------------------------

spk = struct('X',X, 'r',spk_r, 's',spk_s, 'X_cov',X_cov, 'ErrXt',ErrXt );

end
