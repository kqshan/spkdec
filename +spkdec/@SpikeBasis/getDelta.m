function delta = getDelta(self, convT_y)
% Return the improvement in squared approximation error from adding a spike
%   delta = getDelta(self, convT_y)
%
% Returns:
%   delta     [R x T] improvement in squared error from adding a spike at each
%             (sub-sample shift index, time index)
% Required argumens:
%   convT_y   [T x K x R x C] output of self.convT(y)

% Dimensions
[T, K, R, C] = size(convT_y);
assert(K==self.K && R==self.R && C==self.C, self.errid_dim, ...
    'y must be [T x K x R x C]');

% The nice thing about minimizing the squared error is that if
%   x = argmin ||y - A*x||
% then the inner product
%   <A*x, y - A*x> == 0
% which means that
%   ||A*x||^2 + ||y - A*x||^2 = ||y||^2
% Rearranging this, we find that the improvement in squared error
%   delta = ||y||^2 - ||y-A*x||^2 = ||A*x||^2

% So now let's consider solving for x. This is simply a least-squares problem:
%   x = inv(A'*A) * (A'*y)
% We're looking at a single time step, so A is a [(L+W-1)*C x K*C] matrix. Since
% L+W-1 >> K, it can be convenient to come up with a square matrix H such that
%   H'*H == A'*A
% so that we can write
%   x = inv(H'*H) * (A'*y)
%     = inv(H) * inv(H') * (A'*y)
% which means that
%   ||A*x|| = ||H*x|| = ||H' \ (A'*y)||

% Why not use a QR decomposition of A? Well, we need to evaluate this for a
% whole bunch of possible time steps, and we can do that quite efficiently for
% A'*y using our convolution object. If we performed a QR decomposition of A,
% the resulting Q matrix will not have the same channel-specific structure as A,
% and we won't be able to compute its convolution as efficiently.

% Compute each sub-sample shift index separately
H_0 = self.get_gram_chol();
delta = zeros(R, T, 'like',convT_y);
for r = 1:R
    % Extract and reshape the relevant data
    Aty_r = convT_y(:,:,r,:);           % [T x K x 1 x C]
    Aty_r = reshape(Aty_r, [T, K*C]);   % [T x K*C]
    % Solve for H*x = H'\Aty
    % Except a bunch of things are transposed, and se we actually want
    %   Hx = (H*x).' = (H' \ Aty_r.').' = Aty_r / conj(H)
    H = H_0(:,:,r);
    Hx = Aty_r / conj(H);               % [T x K*C]
    % Evaluate its norm
    delta(r,:) = sum(abs(Hx).^2, 2)';
end

end
