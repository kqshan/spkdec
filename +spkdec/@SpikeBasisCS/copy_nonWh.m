function basis_nonWh = copy_nonWh(self)
% Create a copy of this SpikeBasisCS without the whitening operation
%   basis_nonWh = copy_nonWh(self)
%
% Returns:
%   basis_nonWh     SpikeBasisCS without whitening

basis_nonWh = spkdec.SpikeBasisCS(self.basis_cs, 't0',self.t0, ...
    'whitener',spkdec.Whitener.no_whiten(self.C), 'interp',self.interp);

end
