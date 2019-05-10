function addSpikes(self, new_spk, varargin)
% Add new spikes to this Spikes object, keeping self.t in sorted order
%   addSpikes(self, new_spk, [keep_X])
%
% Required arguments:
%   new_spk     Spikes object with new spikes to add
%
% If self and new_spk both have their features defined, then self.X will be
% updated with the new spikes. Otherwise, self.X will be set to [].

% Sort the new spikes into the existing ones
tr = [[self.t; new_spk.t], [self.r; new_spk.r]];
[tr, sortidx] = sortrows(tr);

% Update the spike features
[~,N1] = size(self.X);
[~,N2] = size(new_spk.X);
if (N1==self.N) && (N2==new_spk.N)
    % Both Spikes objects have features defined
    X = [self.X, new_spk.X];
    X = X(:,sortidx);
else
    X = [];
end

% Update self
self.t = tr(:,1);
self.r = tr(:,2);
self.X = X;

end
