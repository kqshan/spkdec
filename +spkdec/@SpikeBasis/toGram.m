function gram = toGram(self)
% Return a spkdec.Gramians object that can be used to compute Gram matrices
%   gram = toGram(self)

% Use the cache if available
gram = self.gramians;
if ~isempty(gram), return; end

% Get the kernels
kernels = self.toKern();                        % [Lw x C x D x R]

% Construct the Gramians object
gram = spkdec.Gramians(kernels);

% Cache this result
self.gramians = gram;

end
