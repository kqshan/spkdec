function A = start_optimization(self, data, prm)
% Initialize the optimization routine based on the given data
%   A = start_optimization(self, data, prm)
%
% Returns:
%   A           [L*C x D] whitened spike basis in Q1 coordinates
% Required arguments:
%   data        [L+W-1 x C x N] detected spike waveforms (whitened)
%   prm         Struct of additional parameters, with fields
%     lambda      Proximal regularizer weight
%     basis_prev  Previous basis (for proximal regularizer)
%     D           Number of basis waveforms overall (if basis_prev not given)
%     zero_pad    [pre,post] #samples of zero-padding to add
%     spk_r       [N x 1] sub-sample shift for each spike (may be empty)
%
% This also defines the following protected properties:
%   Y           [L*C x N x S] spike data in Q1 coordinates, with the S
%               dimension corresponding to shifts of (-dt_search:dt_search)
%   A0          [L*C x D] whitened previous basis in Q1 coordinates
%   lambda      Weight applied to the proximal regularizer ||A-A0||
%   spk_r       User-specified sub-sample shift for each spike

% Convert the given spikes into Q1 coordinates
self.Y = self.convert_spikes_to_Y(data, prm.zero_pad);

% Get a starting spike basis
if isempty(prm.basis_prev)
    assert(prm.lambda==0, self.errid_arg, ...
        'basis_prev must be specified if lambda ~= 0');
    assert(~isempty(prm.D), self.errid_arg, ...
        'D must be specified if basis_prev is not given');
    % Initialize this based on the data
    A = self.init_spkbasis(prm.D);
else
    A = self.convert_spkbasis_to_A(prm.basis_prev);
end
% Store this (and lambda) in our object-level cache
self.A0 = A;
self.lambda = prm.lambda;
% Also save the user-specified spk_r, if given
self.spk_r = prm.spk_r(:);
if ~isempty(self.spk_r) && (self.dt_search > 0)
    warning([self.errid_pfx ':WeirdSearch'], ['The sub-sample shift is ' ...
        'fixed by the user-specified spk_r,\nbut we are still searching '...
        'over full-sample shifts since dt_search > 0. This is kinda weird']);
end

end
