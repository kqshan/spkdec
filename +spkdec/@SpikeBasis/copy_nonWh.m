function basis_nonWh = copy_nonWh(self)
% Create a copy of this SpikeBasis without the whitening operation
%   basis_nonWh = copy_nonWh(self)
%
% Returns:
%   basis_nonWh     SpikeBasis without whitening

basis_nonWh = spkdec.SpikeBasis(self.basis, 't0',self.t0, ...
    'whitener',spkdec.Whitener.no_whiten(self.C), 'interp',self.interp);

end
