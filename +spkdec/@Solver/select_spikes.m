function new_spk = select_spikes(self, delta, old_spk)
% Select spike times given the estimated improvement in reconstruction error
%   new_spk = select_spikes(self, delta, old_spk)
%
% Returns:
%   new_spk     Spike times (t,r) of new spikes (Spikes object)
% Required argumens:
%   delta       [R x T] change in ||A*x-b||^2 from adding a spike at each (r,t)
%   old_spk     Spike times of existing spikes (Spikes object) for det_refrac
%
% This assumes that the sub-sample shifts are ordered so that delta(:) is sorted
% in chronological order. This assumption is satisified by Interpolators made
% using spkdec.Interpolator.make_interp()

% Collapse this into a 1-D problem by assuming that the sub-sample shifts are in
% chronological order
[R,T] = size(delta);
delta = reshape(delta, [R*T, 1]);

% Find threshold-exceeding local maxima
delta_delta = diff(delta);
is_local_max = [true; delta_delta > 0] & [delta_delta < 0; true];
new_t = find(is_local_max & (delta > self.beta));
new_t = gather(new_t);

% Remove any spikes that violate the refractory period
refrac_dt = round(R * self.det_refrac);
if (refrac_dt > 0) && (old_spk.N > 0)
    % Determine which local maxima lie within the refractory period
    old_t = [old_spk.r + R*(old_spk.t-1); Inf];
    [~,last_old_spk_lt_t1] = histc(new_t-refrac_dt-0.5, old_t);
    [~,last_old_spk_le_t2] = histc(new_t+refrac_dt, old_t);
    is_in_refrac = last_old_spk_le_t2 > last_old_spk_lt_t1;
    % Remove these
    new_t = new_t(~is_in_refrac);
end

% Keep only those that are regional maxima in a radius of +/- select_dt
dt = max(R * self.select_dt, refrac_dt);
mask = spkdec.Math.is_reg_max(new_t, delta, dt);
new_t = new_t(mask);
assert(all(diff(new_t) > dt));

% Convert this back to (r,t)
t = ceil(new_t/R);
r = new_t - R*(t-1);
new_spk = spkdec.Spikes(t, r);

end
