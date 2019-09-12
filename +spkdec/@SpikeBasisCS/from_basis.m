function obj = from_basis(basis)
% Construct a SpikeBasisCS object from a non-channel-specific SpikeBasis object
%   obj = SpikeBasisCS.from_basis(basis)
%
% Returns:
%   obj     New SpikeBasisCS object that is functionally identical to `basis`
% Required arguments:
%   basis   SpikeBasis object that has channel-specific basis waveforms

errid_arg = spkdec.SpikeBasisCS.errid_arg;

% Check the dimensions
L = basis.L; C = basis.C; D = basis.D;
assert(mod(D,C)==0, errid_arg, ...
    'Constructing a channel-specific basis requires that D be divisible by C');
K = D/C;

% Make sure the basis is channel-specific to begin with
basis_given = reshape(basis.basis, [L C K C]);
is_zero = reshape(all(all(basis_given==0,1),3), [C C]);
on_diag = logical(eye(C));
assert(all(is_zero | on_diag,'all'), errid_arg, ...
    ['The given basis must be channel-specific, i.e.\n' ...
     'basis(:,c,d)==0 unless ceil(d/K)==c, where K = D/C.']);

% Extract the channel-specific basis
basis_cs = zeros(L, K, C);
for c = 1:C
    basis_cs(:,:,c) = reshape(basis_given(:,c,:,c), [L K]);
end

% Construct the object
obj = spkdec.SpikeBasisCS(basis_cs, 'whitener',basis.whitener, ...
    't0',basis.t0, 'interp',basis.interp);

end
