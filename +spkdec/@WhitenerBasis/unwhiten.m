function [Y_raw, norms] = unwhiten(self, Y_wh)
% Return the raw waveform that minimizes the whitened error to the given signal
%   [Y_raw, norms] = unwhiten(self, Y_wh)
%
% Returns:
%   Y_raw       [L x C x N] unwhitened data
%   norms       [N x 1] norm of each unwhitened waveform (in whitened space)
% Required arguments:
%   Y_wh        [L+W-1 x C x N] data (e.g. spike residual) in whitened space
%
% This finds the "unwhitened" spikes that most closely approximate the given
% whitened spikes. The reason for doing this is that we often have L+W-1 >> L,
% and so Y_wh contains a lot of extraneous information (specifically, Y_wh has
% a large subspace that is orthogonal to what can be represented using a spike
% waveform of length L). However, simply truncating the spikes to a window of
% length L ends up discarding information that we actually need.
%
% Unwhitening addressses this problem by solving for the waveform of length L
% that captures as much of the data as possible (in terms of the distance in
% whitened space). Specifically, this solves
%     minimize  ||Y_wh(:,:,n) - whiten(Y_raw(:,:,n))||

% There's really two avenues for non-destructive dimensionality reduction:
% * Express the spikes in terms of Q1, an orthonormal basis for the span of the
%   whitening operation. Unfortunately, these coordinates are not very portable
%   or interpretable, since they depend not only on the whitener but also the
%   decomposition method (e.g. 'svd' or 'qr').
% * Find the raw spikes that most closely approximate the given data after
%   whitening. This is easier to interpret and doesn't tie us to a specific Q1
%   basis, but finding these raw spikes requires solving a linear inverse 
%   problem that may be poorly conditioned.
% So we are going to go with the latter option here. The max_cond option
% mitigates the numerical conditioning issue by regularizing the problem until
% it is on longer poorly-conditioned.

% Check dimensions
[Lw, C, N] = size(Y_wh);
L = self.L;
assert(Lw==L+self.W-1 && C==self.C, self.errid_dim, ...
    'Y_wh must be [L+W-1 x C x N]');

% Here's our max_cond thing
max_cond = self.max_cond;
    function X = apply_max_cond(X)
        if cond(X) > max_cond
            [U,S,V] = svd(X);
            S = diag(max(diag(S), S(1)/max_cond));
            X = U*S*V';
        end
    end

% This is a pretty straightforward application of the QR decomposition:
%   Y_raw = (wh_00'*wh_00) \ (wh_00'*Y_wh)
%         = (wh_01'*wh_01) \ (wh_01' * Q1' * Y_wh)
%         = wh_01 \ Q1' * Y_wh
% And if spk_r is given, then we replace wh_01 with the appropriate wh_01r

% Compute Z = Q1' * Y_wh
Z = reshape(self.Q1, [Lw*C,L*C])' * reshape(Y_wh,[Lw*C,N]);     % [L*C x N]
if nargout >= 2
    norms = vecnorm(Z,2,1)';
end

% Invert wh_01
wh = reshape(self.wh_01, [L*C, L*C]);
wh = apply_max_cond(wh);
Y_raw = wh \ Z;

% Reshape into the desired output
Y_raw = reshape(Y_raw, [L C N]);

end
