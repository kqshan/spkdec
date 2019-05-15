function basis_mod = copy_modify(self, newbasis)
% Create a copy with a modified basis (but the same whitening and interp)
%   basis_mod = copy_modify(self, newbasis)
%
% Returns:
%   basis_mod   New SpikeBasis object
% Required arguments:
%   newbasis    [L x K x C] new basis waveforms

[L,~,C] = size(newbasis);
assert(L==self.L && C==self.C, self.errid_dim, ...
    'newbasis must have the same L and C as the current one');

basis_mod = spkdec.SpikeBasis(newbasis, 'whitener',self.whitener, ...
    't0',self.t0, 'interp',self.interp);

end
