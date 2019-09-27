function live_init(self, A, prm)
% Initialize the live update of the fitted basis waveforms
%   live_init(self, A, prm)
%
% Required arguments:
%   A           [L*C x D] current basis in Q1 coordinates
%   prm         Struct with fields:
%     live_ah     Axes handle for plotting in, or [] for none
%     live_delay  Animation delay (s) after a live update  

ah = prm.live_ah;
if ~isempty(ah) && ishandle(ah)
    % Determine the inter-channel spacing
    basis = self.convert_A_to_spkbasis(A);
    [L,C,D] = size(basis); %#ok<ASGLU>
    ch_gap = max(0,basis(:,2:end,:)) - min(0,basis(:,1:end-1,:));
    max_gap = max([0; ch_gap(:)]);
    spacing = 1.1 * max_gap;
    % Draw the initial waveforms
    self.live_space = spacing;
    [x,y] = self.convert_A_to_plot_coords(A);
    lh = plot(ah, x-0.5, y, '-');
    set(ah, 'XLim',[0.5 D+0.5], 'XTick',1:D, ...
        'YLim',[0, C+1], 'YTick',1:C, 'YGrid','on');
    xlabel('Feature space dimensions'); ylabel('Data channels');
    drawnow();
    % Update the object-level cache
    self.live_lh = lh;
    self.live_delay = prm.live_delay;
else
    % Make sure that live updates are disabled
    self.live_lh = [];
end

end
