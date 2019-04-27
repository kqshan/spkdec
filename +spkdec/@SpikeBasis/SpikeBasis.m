% Spike basis waveforms, with options for whitening and sub-sample interpolation
%
% SpikeBasis properties (read-only):
% Dimensions
%   C           - Number of data channels
%   K           - Number of spike basis waveforms per channel
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
%   make_basis  - [Static] Initialize a SpikeBasis from detected spikes
% Data conversions
%   toConv      - Return a Convolver object for the whitened basis
%   toKern      - Return a matrix of the whitened basis waveforms
%   toGram      - Return a Gramians object (dot products of the spike basis)
%   toWhBasis   - Return a WhitenerBasis for this whitener and interp
% Convolution
%   conv        - Perform the forward convolution
%   conv_sp     - Perform the forward convolution with a sparse input
%   convT       - Perform the transpose convolution
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
    K
    
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
    function val = get.L(self), val = size(self.basis,1);  end
    function val = get.W(self), val = self.whitener.W;     end
    function val = get.V(self), val = self.L-1 + self.W-1; end
    function val = get.R(self), val = self.interp.R;       end
end

properties (SetAccess=protected)
    % [L x K x C] spike basis waveforms (unwhitened)
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
        % Populate protected caches
        obj.populate_caches();
    end
    
    
    % Data conversions
    
    kern = toKern(self, varargin);
    conv = toConv(self);
    gram = toGram(self);
    whbasis = toWhBasis(self, varargin);
    
    % Convolution
    
    y = conv(self, x);
    y = conv_sp(self, spk, T);
    x = convT(self, y);
    
    % High-level operations
    
    delta = getDelta(self, convT_y);
    spk_X = solve(self, convT_y, spk_t, spk_r, varargin);
    
end

methods (Static)
    % Initialization helper
    obj = make_basis(spikes, K, varargin);
end

% ----------------------     Copy and serialization     ------------------------

methods (Access=protected)
    function obj = copyElement(self)
        obj = copyElement@matlab.mixin.Copyable(self);
        obj.whitener = copy(self.whitener);
        obj.interp = copy(self.interp);
        obj.populate_caches(); % Clear caches
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
        obj = spkdec.SpikeBasis(s.basis, rmfield(s,'basis'));
    end
end


% ------------------------------------------------------------------------------
% ===========================      Protected     ===============================
% ------------------------------------------------------------------------------

% Caches
properties (Access=protected)
    % Convolver object to use for convolution operations
    %
    % This has K*R kernels per channel, so convolver.kernels should be seen as a
    % a [L x K x R x C] matrix that has been reshaped to [L x K*R x C].
    convolver
    
    % Gramians object to use in solve()
    gramians
    
    % [K*C x K*C x R] upper Cholesky decompositions of lag 0 Gram matrices to
    % use in getDelta()
    H_0
end
methods (Access=protected)
    function populate_caches(self)
        % Populate the protected caches
        %   populate_caches(self)
        self.convolver = self.toConv();
        self.gramians = self.toGram();
        self.H_0 = self.compute_H0(self.gramians);
    end
end
methods (Static, Access=protected)
    function H0 = compute_H0(gramians)
        % Compute the Cholesky decomposition of the Gram matrices at lag 0
        %   H0 = compute_H0(gramians)
        %
        % Returns:
        %   H0          [K*C x K*C x R] upper Cholesky decompositions
        % Required arguments:
        %   gramians    Gramians object
        KC = gramians.D; R = gramians.R;
        H0 = zeros(KC, KC, R);
        for r = 1:R
            G = gramians.getGram(0, r, r);
            H0(:,:,r) = chol(G, 'upper');
        end
    end
end


end
