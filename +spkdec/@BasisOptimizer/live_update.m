function live_update(self, A)
% Provide a live update of the current spike waveforms
%   live_update(self, A)
%
% Required arguments:
%   A           [L*C x D] current basis in Q1 coordinates
%
% This also uses self.live_lh and self.live_delay

lh = self.live_lh;
if ~isempty(lh) && all(ishandle(lh))
    % Convert the basis into plot coordinates
    [~,y] = self.convert_A_to_plot_coords(A);
    % Update the y-coords (the x should be constant)
    D = length(lh);
    for d = 1:D, lh(d).YData = y(:,d); end
    % Refresh
    drawnow();
    pause(self.live_delay);
end

end
