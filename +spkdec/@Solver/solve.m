function [spk, resid] = solve(self, A, b, beta)
% Solve the sparse deconvolution problem for the given kernels and data
%   [spk, resid] = solve(self, A, b, beta)
%
% Returns:
%   spk     Detected spikes (Spikes object)
%   resid   [T+V x C] residual: b - A.conv_spk(spk)
% Required arguments:
%   A       Convolution kernels (SpikeBasis object)
%   b       [T+V x C] whitened data to deconvolve
%   beta    Cost of each additional nonzero spike
%
% This uses orthogonal matching pursuit to solve the optimization problem
%   minimize  ||b - A.conv(spk)||^2 + beta*spk.N

% Initialize the loop
self.consts_init(A,b,beta); % Populate cache of problem-specific constants
self.verbose_init();        % Start the verbose output
spk = spkdec.Spikes();      % Start with no spikes detected
resid = self.b;             % No spikes ==> residual is same as given data

% Perform the first round of spike detection
delta = self.eval_improvement(resid);
new_spk = self.select_spikes(delta, spk);

% Iterate until no more spikes are detected
iter = 0;
while (new_spk.N > 0)
    % Append the new spikes and solve for their features
    spk.addSpikes(new_spk);
    spk_X = self.find_spike_features(spk);
    spk.setFeat(spk_X);
    
    % Update the residual
    resid = self.compute_residual(spk);
    
    % Verbose output
    iter = iter + 1;
    self.verbose_update(iter, spk, resid);
    
    % Detect new spikes
    delta = self.eval_improvement(resid);
    new_spk = self.select_spikes(delta, spk);
end

% Clean up
self.verbose_cleanup(iter, spk, resid);
self.consts_cleanup();

end
