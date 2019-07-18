function spk_X = find_spike_features(self, spk)
% Solve the least-squares problem given a set of spike times
%   spk_X = find_spike_features(self, spk)
%
% Returns:
%   spk_X   [D x N] optimal spikes in feature space
% Required arguments:
%   spk     Spike times (Spikes object)

spk_X = self.A.solve( self.At_b, spk.t, spk.r, 'thresh',self.gram_thresh );

end
