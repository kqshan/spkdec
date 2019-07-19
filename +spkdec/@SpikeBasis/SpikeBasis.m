% Spike basis waveforms, with options for whitening and sub-sample interpolation
%
% SpikeBasis properties (read-only):
% Dimensions
%   C           - Number of data channels
%   K           - Number of spike basis waveforms per channel
%   D           - Number of spike basis waveforms overall
%   L           - Basis waveform length (#samples) before whitening
%   W           - Frequency-whitening filter length (#samples)
%   V           - Overall overlap length (#samples), V = L-1 + W-1
%   R           - Sub-sample interpolation ratio (1 = no interpolation)
% Data
%   basis       - [L x K x C] spike basis waveforms (unwhitened)
%   t0          - Sample index (1..L) corresponding to t=0
%   whitener    - Whitener object describing the whitening
%   interp      - Interpolator object describing sub-sample shifts
%
% SpikeBasis methods:
% Construction
%   SpikeBasis  - Construct a new SpikeBasis object
%   copy_modify - Create a copy with a modified basis
%   copy_nonWh  - Create a copy of this basis without whitening
% Data conversions
%   toConv      - Return a Convolver object for the whitened basis
%   toKern      - Return a matrix of the whitened basis waveforms
%   toGram      - Return a Gramians object (dot products of the spike basis)
%   toWhBasis   - Return a WhitenerBasis for this whitener and interp
% Convolution
%   conv        - Perform the forward convolution
%   conv_spk    - Perform the forward convolution with a Spikes object
%   convT       - Perform the transpose convolution
% Spike manipulation
%   reconst     - Reconstruct the waveforms of detected spikes
%   spkNorms    - Return the whitened norms of detected spikes
% High-level operations
%   getDelta    - Return the improvement in squared error from adding a spike
%   solve       - Solve for the spike features given the spike times
% Object management
%   copy        - Create a deep copy of this handle object
%   saveobj     - Serialize a SpikeBasis object to struct
%   loadobj     - [Static] Deserialize a SpikeBasis object from a struct

classdef SpikeBasis < matlab.mixin.Copyable

% ------------------------------------------------------------------------------
% ========================     Public properties     ===========================
% ------------------------------------------------------------------------------


properties (SetAccess=private, Dependent)
    % Number of data channels
    C
    
    % Number of spike basis waveforms per channel
    % The total number of kernels is K*C
    % See also: spkdec.SpikeBasis.D
    K
    
    % Number of spike basis waveforms overall
    D
    
    % Basis waveform length (#samples) before whitening
    % All basis waveforms have the same length.
    L
    
    % Frequency-whitening filter length (#samples)
    W
    
    % Overall overlap length (#samples). V = L-1 + W-1
    V
    
    % Sub-sample interpolation ratio (1 = no interpolation)
    R
end
methods
    function val = get.C(self), val = size(self.basis,3);  end
    function val = get.K(self), val = size(self.basis,2);  end
    function val = get.D(self), val = self.K * self.C;     end
    function val = get.L(self), val = size(self.basis,1);  end
    function val = get.W(self), val = self.whitener.W;     end
    function val = get.V(self), val = self.L-1 + self.W-1; end
    function val = get.R(self), val = self.interp.R;       end
end

properties (SetAccess=protected)
    % [L x K x C] spike basis waveforms (unwhitened)
    %
    % Basis waveforms are specific to a particular channel. This allows for more
    % efficient convolution and makes the feature space easier to interpret.
    basis
    
    % Sample index (1..L) corresponding to t=0
    t0
    
    % spkdec.Whitener object describing the whitening
    %
    % This object contains the frequency-whitening filter and the cross-channel
    % whitening transform used to convert the raw data (and raw waveform basis)
    % into the whitened space in which the approximation error is evaluated. In
    % this sense, this can also be seen as defining a whitened inner product.
    %
    % If no whitener is specified during construction, this is initialized using
    % spkdec.Whitener.no_whiten()
    % See also: spkdec.Whitener
    whitener
    
    % spkdec.Interpolator object describing the sub-sample shifts
    %
    % This object contains transformation matrices that implement a sub-sample
    % shift of the basis waveforms. This allows us to detect spikes at a finer
    % temporal resolution than the source data, which helps reduce the amount of
    % spike variability (by removing spike alignment jitter as a source of
    % variability).
    %
    % If no interp is specified during construction, this is initialized using
    % spkdec.Interpolator.no_interp()
    % See also: spkdec.Interpolator
    interp
end

properties (Constant, Access=private)
    errid_arg = 'spkdec:SpikeBasis:BadArg';
    errid_dim = 'spkdec:SpikeBasis:DimMismatch';
end


% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------


methods
    function obj = SpikeBasis(basis, varargin)
        % SpikeBasis constructor
        %   obj = SpikeBasis(basis, ...)
        %
        % Required arguments:
        %   basis       [L x K x C] spike basis waveforms
        % Optional parameters (key/value pairs) [default]:
        %   t0          Sample index (1..L) for t=0     [ 1 ]
        %   whitener    Whitener object                 [ none ]
        %   interp      Interpolator object             [ none ]
        [L,~,C] = size(basis);
        ip = inputParser();
        ip.addParameter('t0', 1, @isscalar);
        ip.addParameter('whitener', [], ...
            @(x) isempty(x) || isa(x,'spkdec.Whitener'));
        ip.addParameter('interp', [], ...
            @(x) isempty(x) || isa(x,'spkdec.Interpolator'));
        ip.parse( varargin{:} );
        prm = ip.Results;
        % Default whitener
        whitener = prm.whitener;
        if isempty(whitener)
            whitener = spkdec.Whitener.no_whiten(C);
        else
            assert(whitener.C==C, obj.errid_dim, ...
                'whitener.C must match the [L x K x C] basis');
        end
        % Default interpolator
        interp = prm.interp;
        if isempty(interp)
            interp = spkdec.Interpolator.no_interp(L);
        else
            assert(interp.L==L, obj.errid_dim, ...
                'interp.L must match the [L x K x C] basis');
        end
        % Assign values
        obj.basis = gather(double(basis));
        obj.t0 = prm.t0;
        obj.whitener = whitener;
        obj.interp = interp;
    end
    
    % Various forms of copy constructors
    basis_mod = copy_modify(self, newbasis);
    basis_nonWh = copy_nonWh(self);
    
    % Data conversions
    kern = toKern(self, varargin);
    conv = toConv(self);
    gram = toGram(self);
    whbasis = toWhBasis(self);
    
    % Convolution
    y = conv(self, x);
    y = conv_spk(self, spk, T);
    x = convT(self, y);
    
    % Spike manipulation
    spikes = reconst(self, spk, varargin);
    norms = spkNorms(self, spk);
    
    % High-level operations
    delta = getDelta(self, convT_y);
    spk_X = solve(self, convT_y, spk_t, spk_r, varargin);
end

methods (Static)
    % Unit tests
    test(varargin);
end

% ----------------------     Copy and serialization     ------------------------

methods (Access=protected)
    function obj = copyElement(self)
        obj = copyElement@matlab.mixin.Copyable(self);
        for fn = {'whitener','interp','convolver','gramians','whbasis'}
            if ~isempty(self.(fn{1})), obj.(fn{1}) = copy(self.(fn{1})); end
        end
    end
end
methods
    function s = saveobj(self)
        s = struct();
        for fn = {'basis','t0','whitener','interp'}
            s.(fn{1}) = self.(fn{1});
        end
        s.whitener = s.whitener.saveobj();
        s.interp = s.interp.saveobj();
    end
end
methods (Static)
    function obj = loadobj(s)
        s.whitener = spkdec.Whitener.loadobj(s.whitener);
        s.interp = spkdec.Interpolator.loadobj(s.interp);
        obj = spkdec.SpikeBasis(s.basis, rmfield(s,'basis'));
    end
end


% ------------------------------------------------------------------------------
% ===========================      Protected     ===============================
% ------------------------------------------------------------------------------

% Caches
properties (Access=protected)
    % Convolver object, output of toConv() and used in solve()
    %
    % This has K*R kernels per channel, so convolver.kernels should be seen as
    % an [L x K x R x C] matrix that has been reshaped to [L x K*R x C].
    convolver
    
    % Gramians object, output of toGram() and used in solve()
    gramians
    
    % [D x D x R] upper Cholesky decompositions of lag 0 Gram matrices,
    % output of get_gram_chol() and used in getDelta() and spkNorms()
    H_0
    
    % [D x D x R] inverses of H_0
    H_0_inv
    
    % WhitenerBasis object, output of toWhBasis()
    whbasis
end

methods (Access=protected)
    function H0 = get_gram_chol(self)
        % Compute the Cholesky decomposition of the Gram matrices at lag 0
        %   H_0 = get_gram_chol(self)
        %
        % Returns:
        %   H_0     [D x D x R] upper Cholesky decompositions
        %           of self.toGram().getGram(0,r,r)
        H0 = self.H_0;
        if ~isempty(H0), return; end
        gram = self.toGram();
        H0 = zeros(gram.D, gram.D, gram.R);
        for r = 1:gram.R
            H0(:,:,r) = chol(gram.getGram(0,r,r), 'upper');
        end
        self.H_0 = H0;
    end
    
    function H0inv = get_gram_chol_inv(self)
        % Return the inverse of the H_0 matrices returned by get_gram_chol()
        %   H0inv = get_gram_chol_inv(self)
        %
        % Returns:
        %   H0inv   [D x D x R] inverses of self.get_gram_chol()
        H0inv = self.H_0_inv;
        if ~isempty(H0inv), return; end
        H0inv = self.get_gram_chol();
        for r = 1:size(H0inv,3)
            H0inv(:,:,r) = inv(H0inv(:,:,r));
        end
        self.H_0_inv = H0inv;
    end
end

end
