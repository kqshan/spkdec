function norms = spkNorms(self, spk)
% Return the (whitened) norms of the detected spikes
%   norms = spkNorms(self, spk)
%
% Returns:
%   norms       [N x 1] spike 2-norms in whitened space
% Required arguments:
%   spk         Detected spikes (Spikes object)
%
% This is equivalent to:
% >> spikes = self.reconst(spk, 'subshift',true);
% >> norms = squeeze(sqrt(sum(sum(spikes.^2,1),2)));

% H(:,:,r) is the upper Cholesky decomposition of the 0-lag Gramian for
% the sub-sample shift index r, so ||H*X|| = ||self.reconst(X)||
H = self.get_gram_chol();

% Compute the norm for each sub-sample shift
R = self.R; N = spk.N;
norms = zeros(N,1, 'like',spk.X);
if (N==0), return; end % Special case b/c accumarray doesn't like empty inputs
r_idx = accumarray(spk.r, (1:N)', [R 1], @(x) {x});
for r = 1:R
    idx = r_idx{r};
    Z = H(:,:,r) * spk.X(:,idx);
    norms(idx) = vecnorm(Z,2,1)';
end

end
