function gram = toGram(self)
% Return a spkdec.Gramians object that can be used to compute Gram matrices
%   gram = toGram(self)

% Use the cache if available
gram = self.gramians;
if ~isempty(gram), return; end

% Get the kernels
kernels = self.toKern();                        % [Lw x C x K*R x C]

% Permute and reshape so that K*C are together and R is by itself
[Lw, C, ~, ~] = size(kernels);
R = self.R; K = self.K;
kernels = reshape(kernels, [Lw, C, K, R, C]);
kernels = permute(kernels, [1 2 3 5 4]);
kernels = reshape(kernels, [Lw, C, K*C, R]);    % [Lw x C x K*C x R]

% Construct the Gramians object
gram = spkdec.Gramians(kernels);

% Cache this result
self.gramians = gram;

end
