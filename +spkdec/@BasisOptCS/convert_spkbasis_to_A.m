function A = convert_spkbasis_to_A(self, basis)
% Convert the spike basis from raw waveforms into Q2 coordinates
%   A = convert_spkbasis_to_A(self, basis)
%
% Returns:
%   A       [L x K x C] whitened channel-specific spike basis in Q2 coordinates
% Required arguments:
%   basis   [L x C x D] unwhitened spike basis

% Check the dimensions
[L, C, D] = size(basis);
assert(L==self.L, C==self.C, self.errid_dim, ...
    'basis_prev must be [L x C x D] with L=%d and C=%d',self.L,self.C);
assert(mod(D,C)==0, self.errid_arg, ...
    'For this channel-specific basis optimizer, D must be divisible by C');
K = D/C;

% Make sure the basis is block diagonal to begin with
basis_bd = reshape(basis, [L C K C]);
is_zero = reshape(all(all(basis_bd==0,1),3), [C C]);
on_diag = logical(eye(C));
assert(all(is_zero | on_diag,'all'), self.errid_arg, ...
    ['For this channel-specific basis optimizer, basis_prev must be block\n' ...
     'diagonal, i.e. basis(:,c,d)==0 unless ceil(d/K)==c, where K = D/C.']);

% A = wh_02 * basis, applied to each channel
A = zeros(L, K, C);
for c = 1:C
    A(:,:,c) = self.whbasis.wh_02(:,:,c) * reshape(basis_bd(:,c,:,c), [L K]);
end

end
