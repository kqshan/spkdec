function whbasis = toWhBasis(self)
% Return a spkdec.WhitenerBasis object for this whitener and interpolator
%   whbasis = toWhBasis(self)
%
% Returns:
%   whbasis     WhitenerBasis constructed from self.whitener and self.interp

% Use cache if available
whbasis = self.whbasis;
if ~isempty(whbasis), return; end

% Construct a new WhitenerBasis object
whbasis = spkdec.WhitenerBasis( self.whitener, 'interp',self.interp);

% Cache the result
self.whbasis = whbasis;

end
