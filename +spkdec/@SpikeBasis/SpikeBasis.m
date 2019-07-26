% Spike basis waveforms, with options for whitening and sub-sample interpolation
%
% SpikeBasis properties (read-only):
% Dimensions
%   L           - Basis waveform length (#samples) before whitening
%   C           - Number of data channels
%   D           - Number of spike basis waveforms
%   W           - Frequency-whitening filter length (#samples)
%   V           - Overall overlap length (#samples), V = L-1 + W-1
%   R           - Sub-sample interpolation ratio (1 = no interpolation)
% Data
%   basis       - [L x C x D] spike basis waveforms (unwhitened)
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
%   plot        - Plot these spike basis waveforms
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
    % Basis waveform length (#samples) before whitening
    % All basis waveforms have the same length.
    L
    
    % Number of data channels
    C
    
    % Number of spike basis waveforms overall
    D
    
    % Frequency-whitening filter length (#samples)
    W
    
    % Overall overlap length (#samples). V = L-1 + W-1
    V
    
    % Sub-sample interpolation ratio (1 = no interpolation)
    R
end
methods
    function val = get.L(self), val = size(self.basis,1);  end
    function val = get.C(self), val = size(self.basis,2);  end
    function val = get.D(self), val = size(self.basis,3);  end
    function val = get.W(self), val = self.whitener.W;     end
    function val = get.V(self), val = self.L-1 + self.W-1; end
    function val = get.R(self), val = self.interp.R;       end
end

properties (SetAccess=protected)
    % [L x C x D] spike basis waveforms (unwhitened)
    %
    % Each of the D spike basis waveforms is represented as [L x C] waveforms on
    % each channel. If all waveforms are restricted to a single channel, and
    % there is the same number of waveforms for each channel, then consider
    % using the channel-specific subclass spkdec.SpikeBasisCS instead.
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

properties (Constant, Hidden)
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
        %   basis       [L x C x D] spike basis waveforms
        % Optional parameters (key/value pairs) [default]:
        %   t0          Sample index (1..L) for t=0     [ 1 ]
        %   whitener    Whitener object                 [ none ]
        %   interp      Interpolator object             [ none ]
        [L,C,~] = size(basis);
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
    plot(self, varargin);
    kern = toKern(self, varargin);
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
    % Convolver object, output of toConv() and used in convolution methods
    %
    % This has D*R kernels, and convolver.kernels is [L+W-1 x C x D*R].
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
    function conv = toConv(self)
        % Construct a spkdec.Convolver object for the whitened basis waveforms
        %   conv = toConv(self)
        %
        % Returns:
        %   conv    spkdec.Convolver object with [L+W-1 x C x D*R] kernels
        conv = self.convolver;
        if ~isempty(conv), return; end
        kern = self.toKern();
        wh_t0 = self.t0 + self.whitener.delay;
        conv = spkdec.Convolver(kern(:,:,:), 't0',wh_t0);
        self.convolver = conv;
    end

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
