function whbasis = toWhBasis(self, varargin)
% Return a spkdec.WhitenerBasis object for this whitener and interpolator
%   whbasis = toWhBasis(self, ...)
%
% Returns:
%   whbasis     WhitenerBasis constructed from self.whitener and self.interp
% Optional arguments:
%   ...         Additional arguments are forwarded to WhitenerBasis constructor

whbasis = spkdec.WhitenerBasis( self.whitener, ...
    'interp',self.interp, varargin{:});

end
