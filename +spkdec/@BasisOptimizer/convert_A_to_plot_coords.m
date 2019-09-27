function [x,y] = convert_A_to_plot_coords(self, A)
% Convert the given spike basis from Q1 coordinates into vectors for plotting
%   [x,y] = convert_A_to_plot_coords(self, A)
%
% Returns:
%   x       [(R*L+1)*C x D] x-coordinates for plotting
%   y       [(R*L+1)*C x D] y-coordinates for plotting
% Required arguments:
%   A       [L*C x D] spike basis waveforms in Q1 coordinates
%
% This uses self.live_space to determine the inter-channel spacing for y.

% Get the raw basis waveforms
basis = self.convert_A_to_spkbasis(A);
[L,C,D] = size(basis); %#ok<ASGLU>
% Interpolate
y = self.whbasis.interp.interp(basis(:,:));
[RL,CD] = size(y); assert(CD==C*D);
y = reshape(y, [RL, C, D]);
% Apply the inter-channel spacing
y = y/self.live_space + (1:C);
% Prepend a NaN separator
y = [NaN(1,C,D); y];        % [1+R*L x C x D]
% Create x-coords
x = (0:RL)' / RL;           % [1+R*L x 1]
x = repmat(x, [1 C]);       % [1+R*L x C]
x = x + shiftdim(1:D,-1);   % [1+R*L x C x D]
% Collapse over the C dimension
x = reshape(x, [(1+RL)*C, D]);
y = reshape(y, [(1+RL)*C, D]);

end
