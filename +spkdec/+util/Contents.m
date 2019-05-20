% Utility functions for spike deconvolution
%
% Main routines
%   make_spk_whitener   - Make a new Whitener for spike detection
%   make_spkbasis       - Fit a new SpikeBasis for the given data + Whitener
%   detect_spikes       - Detect spikes using the given SpikeBasis
%
% Helper functions
%   estimate_spectra    - Estimate noise power spectral density
%   estimate_chcov      - Estimate cross-channel covariance of filtered data
%   init_spkbasis       - Initialize SpikeBasis by traditional spike detection
%   update_spkbasis     - Perform a single stochastic gradient descent step
%
% Plotting functions
%   plot_spikes         - Plot the detected spikes (individually and in context)
%   plot_features       - Produce scatterplots for all pairs of feature coords
