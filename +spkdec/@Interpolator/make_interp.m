function [obj, X] = make_interp(L, R, varargin)
% Helper function for making Interpolator objects
%   [obj, interpMat] = make_interp(L, R, ...)
%
% Returns:
%   obj         Interpolator object
%   interpMat   [R*L x L] equivalent matrix performing the interpolation
% Required arguments:
%   L           Signal length (#samples)
%   R           Interpolation ratio (1 = no interpolation)
% Optional parameters (key/value pairs) [default]:
%   method      Interp. method {linear,hermite,spline,sinc}     ['spline']
%
% The 'spline' and 'hermite' methods both use cubic polynomials with a boundary
% condition of f'(x)=0, but differ in how they treat the interior derivatives.
% 'spline' solves a tridiagonal system to produce a continuous second derivative
% (which means that the resulting shift matrices are not exactly Toeplitz). 
% 'hermite' uses the central difference of nearby points (a Catmull-Rom spline).

ip = inputParser();
ip.addParameter('method', 'spline', @ischar);
ip.parse( varargin{:} );
prm = ip.Results;

% Construct the interpolator as a [R*L x L] matrix
switch (prm.method)
    case 'linear'
        % Linear interpolation between y(k) and y(k+1), k = 1..L
        % Set x to be a local interpolator
        t = (1:R)'/R;
        x = [1-t, t];               % [R x 2]
        % Tile this into a big matrix
        X = toeplify(x, L);         % [R*L x L+1]
        % Apply the y(0) = 0 boundary condition
        X = X(:, 2:end);            % [R*L x L]
        
    case 'hermite'
        % Catmull-Rom spline
        % Set x to be an [R x 4] local interpolator
        p0 = [ 0  1  0  0];
        p1 = [ 0  0  1  0];
        m0 = [-1  0  1  0]/2; % Slope = central difference
        m1 = [ 0 -1  0  1]/2;
        x = hermite_to_x(R) * [p0; m0; p1; m1]; % [R x 4]
        % Tile this into a big matrix
        X = toeplify(x, L);         % [R*L x R+3]
        % Apply the y(-1) = y(0) = y(L+1) = 0 boundary condition
        X = X(:, 3:end-1);          % [R*L x L]
        
    case 'sinc'
        % sinc interpolation
        t = (0:R*L-1)'/R;           % [R*L x 1]
        X = toeplitz(sinc(t));      % [R*L x R*L]
        X = X(:, R:R:end);          % [R*L x L]
        
    case 'spline'
        % Spline with continuous second derivative
        % Solve for the knot positions (p) and slopes (m) to satisfy:
        %   p(0)   = 0
        %   p(1:T) = y(1:T)
        %   p(T+1) = 0
        %   m(0)   = 0
        %   m(T+1) = 0
        % And the continuity of the 2nd derivative (C2) comes out to:
        %   m(t-1) + 4*m(t) + m(t+1) = -3*p(t-1) + 3*p(t+1)
        y_to_p = [zeros(1,L); eye(L)];          % [L+1 x L]
        C2_lhs = spdiags(ones(L,1)*[1 4 1], -1:1, L, L);
        C2_rhs = spdiags(ones(L,1)*[-3,3], [-1,1], L, L);
        C2_rhs = full(C2_rhs);
        y_to_m = [zeros(1,L); C2_lhs\C2_rhs];   % [L+1 x L]
        % Then we need to convert from Hermite form to interpolated values
        H = hermite_to_x(R); % [R x 4]
        p_to_x = toeplify(H(:,[1 3]), L);       % [R*L x L+1]
        m_to_x = toeplify(H(:,[2 4]), L);       % [R*L x L+1]
        % And then we just combine the two
        X = p_to_x * y_to_p + m_to_x * y_to_m;  % [R*L x L]
        
    otherwise
        error(spkdec.Interpolator.errid_arg, ...
            'Unsupported interpolation method "%s"', prm.method);
end

% Construct the interpolator
obj = spkdec.Interpolator(X);

end


% -------------------------     Helper functions     ---------------------------


function X = toeplify(x, L)
% Produce a block Toeplitz matrix from the given block
%   X = toeplify(x, L)
%
% Returns:
%   X       [L*N x M+L-1] Toeplitz-ified matrix
% Required arguments:
%   x       [N x M] block to rpeat
%   L       Number of rows to repeat this for
%
% Actually not sure if "block Toeplitz" is the right name for this, but the
% general idea is that for:
%   x = [ 1 2 ]  toeplify(x, 3) = [ 1 2     ] 
%       [ 3 4 ]                   [ 3 4     ]
%                                 [   1 2   ]
%                                 [   3 4   ]
%                                 [     1 2 ]
%                                 [     3 4 ]
[N,M] = size(x);
X = zeros(L*N, M+L-1, 'like',x);
for ii = 1:L
    X((1:N) + N*(ii-1), (1:M) + (ii-1)) = x;
end

end


function H = hermite_to_x(R)
% Return a matrix implementing the transform from Hermite form to spline values
%   H = hermite_to_x(R)
%
% Returns:
%   H     [R x 4] transform from [p0; m0; p1; m1] to values at x=(1:R)'/R
% Required arguments:
%   R     Interpolation ratio
t = (1:R)'/R;
t2 = t.^2;
t3 = t.^3;
H = [2*t3-3*t2+1, t3-2*t2+t, -2*t3+3*t2, t3-t2];
end
