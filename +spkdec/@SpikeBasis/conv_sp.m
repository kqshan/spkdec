function y = conv_sp(self, spk, Ty)
% Perform the forward convolution with a sparse input
%   y = conv_sp(self, spk, T)
%
% Returns:
%   y       [T x C] convolution output
% Required arguments:
%   spk     Struct with fields:
%     t       [N x 1] spike times (1..T)
%     X       [K*C x N] spike features
%   T       Desired output length (#samples)
%
% If the interpolation ratio R > 1, then fractional spike times are allowed:
% * mod(t,1) must be reasonably close to a multiple of 1/R
% * Sub-sample shifts are performed using self.interp.shifts(:,:,R*mod(t,1)+1)
%
% We also expect that (t0+filt_delay) <= spk.t < T-V+t0+filt_delay, where
% filt_delay = self.whitener.delay

% Dimensions
K = self.K;
C = self.C;
R = self.R;
V = self.V;
% Convolution input length (so that the output has length T)
T = Ty - V;
conv_t0 = self.convolver.t0;

% Turn the fractional part of the spike times into sub-sample shift indices
r_given = R*mod(spk.t(:),1) + 1;
r = round(r_given);
assert(all(r <= R) && all(abs(r-r_given) < 0.1/R), self.errid_arg, ...
    'Fractional spike times must be a multiple of 1/R');

% Apply the filter delay and spike offset to the spike times
t_given = floor(spk.t(:));
t = t_given - conv_t0 + 1;
assert(all(t >= 1) && all(t <= T), self.errid_dim, ...
    'spk.t must lie in the range [%d, %d)', conv_t0, T+conv_t0);

% Construct the convolution input
[KC, N] = size(spk.X);
assert(KC==K*C && N==length(t), self.errid_dim, 'spk.X must be [K*C x N]');
x = zeros(K*C, R*T, 'like',spk.X);  % [K*C x R*T]
rt = r + R*(t-1);
x(:,rt) = spk.X;
% Reshape into the desired form
x = reshape(x, [K C R T]);          % [K x C x R x T]
x = permute(x, [4 1 3 2]);          % [T x K x R x C]

% Perform the convolution
y = self.conv(x);

end
