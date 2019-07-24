function compute_decompositions(self)
% Compute the basis decompositions that comprise this object
%   compute_decompositions(self)
%
% Before calling this method, self.whitener, .interp, and .method must be
% assigned. This method then defines the Q1, wh_01, wh_01r, Q2, wh_02, wh_02r,
% map_21, and map_21r properties.

% Collect the relevant inputs into local variables
shifts = self.interp.shifts;            % [L x L x R] shift matrices
[L, L_, R] = size(shifts);
assert(L==L_);
whiten = self.whitener.toMat(L);
[Lw, C, L_, C_] = size(whiten);
assert(C==C_ && L==L_);
whiten = reshape(whiten,[Lw*C, L*C]);   % [Lw x C x L x C] whitening operation
method = self.method;
max_cond = self.max_cond;

% Decompose Q1 * wh_01 = whiten
%   Q1    : [Lw x C x L*C]
%   wh_01 : [L*C x L x C]
switch (method)
    case 'qr'
        [Q, X] = qr(whiten, 0);
    case 'svd'
        [Q, S, V] = svd(whiten, 'econ');
        X = S*V';
    otherwise
        error(self.errid_arg, 'Unsupported method "%s"', method);
end
Q1 = reshape(Q, [Lw, C, L*C]);
wh_01 = reshape(X, [L*C, L, C]);
% Get the shifted versions
%   wh_01r : [L*C x L x C x R]
wh_01r = zeros(L*C, L, C, R);
for r = 1:R
    shift_r = shifts(:,:,r);
    % Apply it to each channel
    for c = 1:C
        wh_01r(:,:,c,r) = wh_01(:,:,c) * shift_r;
    end
end

% Decompose Q2 * wh_02 = whiten
%   Q2    : [Lw x C x L x C]
%   wh_02 : [L x L x C]
whiten_c = reshape(whiten, [Lw*C, L, C]);
Q2 = zeros(Lw*C, L, C);
wh_02 = zeros(L, L, C);
for c = 1:C
    Y = whiten_c(:,:,c); % [Lw*C x L]
    switch (method)
        case 'qr'
            [Q, X] = qr(Y, 0);
        case 'svd'
            [Q, S, V] = svd(Y, 'econ');
            X = S*V';
    end
    Q2(:,:,c) = Q;
    wh_02(:,:,c) = X;
end
Q2 = reshape(Q2, [Lw, C, L, C]);
% Get the shifted versions
%   wh_02r : [L x L x C x R]
wh_02r = zeros(L, L, C, R);
for r = 1:R
    shift_r = shifts(:,:,r);
    for c = 1:C
        wh_02r(:,:,c,r) = wh_02(:,:,c) * shift_r;
    end
end

% Get the map from Q2 to Q1
%   map_21 : [L*C x L x C]
map_21 = reshape(Q1,[Lw*C, L*C])' * reshape(Q2, [Lw*C, L*C]);
map_21 = reshape(map_21, [L*C, L, C]);
% And the shifted versions
%   map_21r : [L*C x L x C x R]
map_21r = zeros(L*C, L, C, R);
for c = 1:C
    % Apply the condition number bound if necessary
    wh_02_c = impose_max_cond(wh_02(:,:,c), max_cond);
    % map_21r = map_21 * wh_02r(:,:,r) / wh_02
    for r = 1:R
        map_21r(:,:,c,r) = map_21(:,:,c) * wh_02r(:,:,c,r) / wh_02_c;
    end
end

% Get the sub-sample shift in Q1 coordinates
%   shift1r : [L*C x L*C x R]
shift1r = zeros(L*C, L*C, R);
% Apply the condition number bound if necessary
wh_01_mat = reshape(wh_01, [L*C, L*C]);
wh_01_mat = impose_max_cond(wh_01_mat, max_cond);
% shift1r = wh_01r(:,:,r) / wh_01
for r = 1:R
    wh_01r_mat = reshape(wh_01r(:,:,:,r), [L*C, L*C]);
    shift1r(:,:,r) = wh_01r_mat / wh_01_mat;
end

% Assign these values to self
self.wh_00  = reshape(whiten, [Lw, C, L, C]);
self.Q1     = Q1;
self.wh_01  = wh_01;
self.wh_01r = wh_01r;
self.shift1r = shift1r;
self.Q2     = Q2;
self.wh_02  = wh_02;
self.wh_02r = wh_02r;
self.map_21 = map_21;
self.map_21r = map_21r;

end


% ------------------------     Helper functions     ----------------------------

function X = impose_max_cond(X, max_cond)
% Impose a bound on the maximum condition number
%   X = impose_max_cond(X, max_cond)
%
% Returns:
%   X           Same matrix with a constraint on the maximum condition number
% Required arguments:
%   X           Square matrix
%   max_cond    Upper bound on the desired condition number
if cond(X) > max_cond
    [U,S,V] = svd(X);
    S = diag(max(diag(S), S(1)/max_cond));
    X = U * S * V';
end
end
