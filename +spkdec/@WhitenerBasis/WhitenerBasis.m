% Orthonormal bases for whitened waveforms with finite support
%
% WhitenerBasis properties (read-only):
% Dimensions
%   C         - Number of data channels
%   W         - Whitening filter length (#samples)
%   L         - Waveform length (#samples) before whitening
%   R         - Interpolation ratio (1 = no interpolation)
% Data
%   whitener  - Whitener object describing the whitening
%   interp    - Interpolator object describing sub-sample shifts
%   method    - Method used to perform these decompositions
%   max_cond  - Maximum condition number (imposed when inverting matrices)
% Bases
%   wh_00     - Whitening operation expressed as a matrix
%   Q1        - Orthonormal basis for the whitener output
%   wh_01     - Whitening operation as a map from raw waveforms to the Q1 basis
%   wh_01r    - Alternate versions of wh_01 with sub-sample shift
%   Q2        - Channelwise orthonormal basis for the whitener output
%   wh_02     - Whitening operation as a map from raw waveforms to the Q2 basis
%   wh_02r    - Alternate versions of wh_02 with sub-sample shift
%   map_21    - Map from the Q2 basis to the Q1 basis that preserves whitening
%   map_21r   - Maps from Q2 to Q1 that preserve whitening + sub-sample shift
%
% WhitenerBasis methods:
%   WhitenerBasis - Constructor
%   whiten    - Whiten a waveform of length L
%   unwhiten  - Return the raw waveform that minimizes the whitened error
% Object management
%   copy      - Create a deep copy of this handle object
%   saveobj   - Serialize an WhitenerBasis object to a struct
%   loadobj   - [Static] Deserialize a WhitenerBasis object from a struct


classdef WhitenerBasis < matlab.mixin.Copyable

% ------------------------------------------------------------------------------
% ========================     Public properties     ===========================
% ------------------------------------------------------------------------------


properties (SetAccess=private, Dependent)
    % Number of data channels
    C
    
    % Whitening filter length (#samples)
    % The overall length of the whitened waveform is therefore L+W-1
    W
    
    % Waveform length (#samples) before whitening
    % This WhitenerBasis object produces explicit bases for the whitened
    % waveforms, which requires us to pre-select the waveform support.
    L
    
    % Interpolation ratio (1 = no interpolation)
    R
end
methods
    function val = get.C(self), val = self.whitener.C; end
    function val = get.W(self), val = self.whitener.W; end
    function val = get.L(self), val = self.interp.L;   end
    function val = get.R(self), val = self.interp.R;   end
end

properties (SetAccess=protected)
    % Whitener object describing the whitening
    whitener
    
    % Interpolator object describing sub-sample shifts
    %
    % If no Interpolator is specified when constructing this WhitenerBasis,
    % this property is initialized using spkdec.Interpolator.no_interp()
    interp
    
    % Method used to perform these decompositions (string)
    %
    % The options are:
    %   qr  - QR decomposition: wh_01 will be upper triangular and map_02 will
    %         be block diagonal with upper triangular blocks
    %   svd - Singular value decomposition: wh_01 will have orthogonal rows 
    %         (wh_01*wh_01' is diagonal and sorted in decreasing order) and 
    %         likewise for the blocks of map_02
    method
    
    % Maximum condition number (imposed when inverting matrices)
    %
    % This is used in the following operations:
    % * Constructing map_21r requires inverting wh_02
    % * Member function unwhiten() requires inverting wh_01
    %
    % In these cases, the matrix being inverted (let's call it X) will be
    % replaced by:
    %   [U,Sigma,V] = svd(X)
    %   sig = diag(Sigma);
    %   X_new = U * diag(max(sig, sig(1)/max_cond)) * V'
    %
    % We typically have cond(wh_02) < cond(wh_01), so it's the unwhiten() method
    % that is more likely to be affected by this.
    max_cond
end

% -----------------------    Computed decompositions     -----------------------

properties (SetAccess=protected)
    % Whitening operation expressed as a matrix
    %
    % [L+W-1 x C x L x C] array, which may be interpreted as:
    % * Linear map [L x C] --> [L+W-1 x C] that maps raw to whitened waveforms
    % * Output of whitener.toMat(L)
    % * Reshaped [(L+W-1)*C x L*C] matrix that will be decomposed into:
    %       wh_00 == Q1 * wh_01
    %             == Q2 * wh_02
    %             == Q1 * map_21 * wh_02
    wh_00
    
    % Orthonormal basis for the output (span) of the whitening operator
    %
    % [L+W-1 x C x L*C] array, which may be interpreted as a:
    % * Set of L*C vectors that span the output of the whitening operator, where
    %   each vector is shaped as an [L+W-1 x C] array, and the Frobenius inner
    %   product <Q1(:,:,a),Q1(:,:,b)> is either 1 (if a==b) or 0 (a~=b)
    % * Linear map [L*C] --> [L+W-1 x C] from coordinates in the Q1 basis to the
    %   original data space
    % * Reshaped [(L+W-1)*C x L*C] orthonormal matrix (Q1'*Q1 == I). In this
    %   form, and with appropriate reshaping of wh_01, we have:
    %       Q1 * wh_01 == whitener.toMat(L, 'flatten',true)
    %
    % The reason for creating this basis is that L+W-1 is often much larger than
    % L, so it is more efficient to represent the whitened residual in this
    % basis.
    Q1
    
    % Whitening operation as a map from raw waveforms to the Q1 basis
    %
    % [L*C x L x C] array, which may be interpreted as a:
    % * Linear map [L x C] --> [L*C] that represents the whitening operation
    %   using the default basis for the input (raw waveforms) and the Q1 basis
    %   for the output.
    % * Reshaped [L*C x L*C] matrix. In this form, and with appropriate
    %   reshaping of Q1, we have:
    %       Q1 * wh_01 == whitener.toMat(L, 'flatten',true)
    %
    % Since Q1 is orthonormal, we can evaluate the whitened inner product
    % between raw waveforms using wh_01. This decomposition is also used in
    % WhitenerBasis.unwhiten()
    wh_01
    
    % Alternate versions (R total) of wh_01 with interpolation
    %
    % [L*C x L x C x R] array, which is just R versions of wh_01 that correspond
    % to the combination of applying the sub-sample shift interp.shifts(:,:,r)
    % and the whitening operator.
    %
    % In the "Reshaped [L*C x L*C] matrix" view of wh_01, we have (for r=1..R):
    %       Q1 * wh_01r(:,:,r) = whitener.toMat(L, 'flatten',true) ...
    %                            * blkdiagify(interp.shifts(:,:,r))
    % where blkdiagify(...) returns a [L*C x L*C] block diagonal matrix with C
    % copies of interp.shifts(:,:,r) along its diagonal.
    wh_01r
    
    % Channelwise orthonormal basis for the output (span) of the whitener
    %
    % [L+W-1 x C x L x C] array, which may be interpreted as:
    % * C bases (one for each data channel), each of which is a:
    %   - Set of L vectors that span the output of the whitening operator when
    %     applied to this channel alone. Each vector is shaped as an [L+W-1 x C]
    %     array, and the Frobenius inner product <Q2(:,:,i,c), Q2(:,:,j,c)> is
    %     either 1 (if i==j) or 0 (i~=j).
    % * Linear map [L x C] --> [L+W-1 x C] from coordinates in the Q2 basis to
    %   the original data space.
    % * C reshaped [(L+W-1)*C x L] matrices, each of which is orthonormal
    %   (Q2(:,:,c)'*Q2(:,:,c) == I). In this form, and with appropriate
    %   reshaping of wh_02, we have:
    %       Q2 * wh_02 == whitener.toMat(L, 'flatten',true)
    %
    % Note that Q2 spans the same linear subspace as Q1. For many operations, Q1
    % is the more convenient basis since it is orthonormal. However, optimizing
    % the kernels involves a "channelwise orthonormal" constraint that is more
    % conveniently specified in the Q2 basis.
    Q2
    
    % Whitening operation as a block-diagonal map from raw waveforms to Q2 basis
    %
    % [L x L x C] array, which may be interpreted as:
    % * Linear map [L x C] --> [L x C], implemented as C independent linear maps
    %   [L] --> [L], that represents the whitening opeartion using the default
    %   basis for the input (raw waveforms) and the Q2 basis for the output.
    % * Packed representation of a [L*C x L*C] block diagonal matrix. In this
    %   form, and with appropriate reshaping of Q2, we have:
    %       Q2 * wh_02 == whitener.toMat(L, 'flatten',true)
    %
    % Since Q2 is channelwise orthonormal, we can use this to help evaluate the
    % constraint that the kernels A be channelwise orthonormal. If we represent
    % A as an [L*C x K*C] block diagonal matrix with C [L x K] blocks, then:
    %   A is channelwise orthonormal <==> wh_02*A is block diagonal with
    %                                     orthonormal blocks
    %
    % Note that wh_02 is typically better-conditioned than wh_01.
    wh_02
    
    % Alternate versions (R total) of wh_02 with interpolation
    %
    % [L x L x C x R] array, which is just R versions of wh_02 that correspond
    % to the combination of applying the sub-sample shift interp.shifts(:,:,r)
    % and the whitening operator.
    %
    % In the "Packed representation of a [L*C x L*C] block diagonal matrix" view
    % of wh_02, we have (for r = 1..R);
    %       Q2 * wh_02r(:,:,r) = whitener.toMat(L, 'flatten',true)
    %                            * blkdiagify(interp.shifts(:,:,r))
    % where blkdiagify(...) returns a [L*C x L*C] block diagonal matrix with C
    % copies of interp.shifts(:,:,r) along its diagonal.
    wh_02r
    
    % Map from the Q2 basis to the Q1 basis that preserves whitening
    %
    % [L*C x L x C] array, which may be interpreted as a:
    % * Linear map [L x C] --> [L*C] that converts a whitened waveform from
    %   coordinates in the Q2 basis to coordinates in the Q1 basis.
    % * Reshaped [L*C x L*C] matrix. In this form, and with appropriate
    %   reshaping of Q1 and wh_02, we have:
    %       Q1 * map_21 * wh_02 == whitener.toMat(L, 'flatten',true)
    %   Note that this is also equivalent to:
    %            map_21 * wh_02 == wh_01
    %
    % When optimizing the kernels, we often need to convert between the Q2 basis
    % (in which the "channelwise orthonormal" constraint is easy to enforce) and
    % the Q1 basis (which is orthonormal and hence preserves the whitened inner
    % product). Note that map_21 need not be invertible.
    map_21
    
    % Maps from the Q2 basis to the Q1 basis that preserve whitening + shift
    %
    % [L*C x L x C x R] array, which may be interpreted as:
    % * R linear maps [L x C] --> [L*C], each of which converts an unshifted, 
    %   whitened waveform in the Q2 basis to a interp.shifts(:,:,r)-shifted,
    %   whitened waveform in the Q1 basis.
    % * R reshaped [L*C x L*C] matrices. In this form, and with appropriate
    %   reshaping of the other matrices, we have:
    %       Q1 * map_21r(:,:,r) * wh_02 == whitener.toMat(L,'flatten',true)
    %                                      * blkdiagify(interp.shifts(:,:,r))
    %   where blkdiagify(...) returns a [L*C x L*C] block diagonal matrix with C
    %   copies of interp.shifts(:,:,r) along its diagonal. Let us introduce a
    %   shorthand shift(r) = blkdiagify(interp.shifts(:,:,r)) and note that:
    %       map_21r(:,:,r) * wh_02 == wh_01 * shift(r)
    %                              == map_21 * wh_02 * shift(r)
    %   Under the assumption that wh_02 is invertible, we have:
    %       map_21r(:,:,r) == map_21 * wh_02 * shift(r) / wh_02
    %
    % Like map_21, this is primarily used during the kernel optimization
    map_21r
end

properties (Constant, Access=private)
    errid_arg = 'spkdec:WhitenerBasis:BadArg';
    errid_dim = 'spkdec:WhitenerBasis:DimMismatch';
end


% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------


methods
    function obj = WhitenerBasis(whitener, varargin)
        % Construct a new WhitenerBasis object
        %   obj = WhitenerBasis(whitener, ...)
        %
        % Required arguments:
        %   whitener  Whitener object
        % Optional parameters (key/value pairs) [default]:
        %   interp    Interpolator object               [ none ]
        %   L         Waveform length (#samples)        [defer to interp]
        %   method    Decomposition method              ['qr']
        %   max_cond  Maximum condition number          [ 300 ]
        %
        % Either 'L' or 'interp' needs to be provided
        ip = inputParser();
        ip.addParameter('interp', []);
        ip.addParameter('L', [], @(x) isempty(x) || isscalar(x));
        ip.addParameter('method', 'qr', @ischar);
        ip.addParameter('max_cond', 300, @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        % Get the interpolator
        interp = prm.interp; L = prm.L;
        if isempty(interp)
            assert(~isempty(L), obj.errid_arg, ...
                'Either the "interp" or "L" param must be specified');
            interp = spkdec.Interpolator.no_interp(L);
        else
            assert(isa(interp,'spkdec.Interpolator'), obj.errid_arg, ...
                'interp must be a spkdec.Interpolator object');
        end
        % Set the values
        assert(isa(whitener,'spkdec.Whitener'), obj.errid_arg, ...
            'whitener must be a spkdec.Whitener object');
        obj.whitener = whitener;
        obj.interp = interp;
        obj.method = prm.method;
        obj.max_cond = prm.max_cond;
        % Compute the decompositions
        obj.compute_decompositions();
    end
    
    % Math
    Y_wh = whiten(self, Y_raw);
    [Y_raw, norms] = unwhiten(self, Y_wh);
end

% ----------------------     Copy and serialization     ------------------------

methods (Access=protected)
    function obj = copyElement(self)
        obj = copyElement@matlab.mixin.Copyable(self);
        obj.whitener = copy(self.whitener);
        obj.interp = copy(self.interp);
    end
end
methods
    function s = saveobj(self)
        s = struct();
        for fn = {'whitener','interp','method','max_cond'}
            s.(fn{1}) = self.(fn{1});
        end
        s.whitener = s.whitener.saveobj();
        s.interp = s.interp.saveobj();
    end
end
methods (Static)
    function obj = loadobj(s)
        obj = spkdec.WhitenerBasis(s.whitener, rmfield(s,'whitener'));
    end
end


% ------------------------------------------------------------------------------
% ====================    Protected properties/methods     =====================
% ------------------------------------------------------------------------------


methods (Access=protected)
    % Compute the decompositions and update the object
    compute_decompositions(self);
end

end
