function G_blkband = get_gram_for_overlaps(self, overlaps)
% Build a block-banded matrix of Gram matrices for the specified overlaps
%   G_blkband = get_gram_for_overlaps(self, overlaps)
%
% Returns:
%   G_blkband   [D x D x B x N] block-banded matrix of Gram matrices
% Required arguments:
%   overlaps    Struct with fields:
%     bands       [B x N] case index (1..P) for each overlap
%     cases       [P x 3] unique values of [lag, r1, r2]
%
% This constrcuts G_blkband so that
%   G_blkband(:,:,b,n) = self.getGram(lag, r1, r2)
%   [lag, r1, r2] = overlaps.cases(overlaps.bands(b,n), :)

% Get the Gram matrices for each case
lrr = overlaps.cases;
G_tiles = self.getGram(lrr(:,1), lrr(:,2), lrr(:,3));   % [D x D x P]

% Constructing G_blkband is then an array indexing problem
[D, ~, P] = size(G_tiles);
G_tiles = reshape(G_tiles, [D*D, P]);       % [D*D x P]
bands = overlaps.bands;                     % [B x N] index (1..P)
[B, N] = size(bands);
G_blkband = G_tiles(:, bands(:));           % [D*D x B*N]
% Reshape into the desired dimensions
G_blkband = reshape(G_blkband, [D D B N]);  % [D x D x B x N]

end
