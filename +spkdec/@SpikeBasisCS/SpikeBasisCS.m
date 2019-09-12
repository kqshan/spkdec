% SpikeBasis subclass for the special case of channel-specific (CS) waveforms
%
% SpikeBasisCS properties (read-only):
% Dimensions
%   L           - Basis waveform length (#samples) before whitening
%   C           - Number of data channels
%   K           - Number of spike basis waveforms per channel (D = K*C)
%   D           - Number of spike basis waveforms overall (D = K*C)
%   W           - Frequency-whitening filter length (#samples)
%   V           - Overall overlap length (#samples), V = L-1 + W-1
%   R           - Sub-sample interpolation ratio (1 = no interpolation)
% Data
%   basis_cs    - [L x K x C] channel-specific basis waveforms
%   basis       - [L x C x D] spike basis waveforms (unwhitened)
%   t0          - Sample index (1..L) corresponding to t=0
%   whitener    - Whitener object describing the whitening
%   interp      - Interpolator object describing sub-sample shifts
%
% SpikeBasisCS methods:
% Construction
%   SpikeBasisCS  - Construct a new SpikeBasisCS object
%   copy_modify   - Create a copy with a different basis (no longer CS)
%   copy_modifyCS - Create a copy with a different channel-specific basis
%   copy_nonWh    - Create a copy of this basis without whitening
%   from_basis    - [Static] Construct from a non-CS SpikeBasis object
% Data conversions
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
%   saveobj     - Serialize a SpikeBasisCS object to struct
%   loadobj     - [Static] Deserialize a SpikeBasisCS object from a struct

classdef SpikeBasisCS < spkdec.SpikeBasis

% ------------------------------------------------------------------------------
% ========================     Public properties     ===========================
% ------------------------------------------------------------------------------


properties (SetAccess=private, Dependent)
    % Number of spike basis waveforms per channel
    %
    % The total number of kernels is D = K*C
    K
end
methods
    function val = get.K(self), val = size(self.basis_cs,2); end
end

properties (SetAccess=protected)
    % [L x K x C] channel-specific basis waveforms
    %
    % This is a more space-efficient way of representing the basis waveforms for
    % this channel-specific subclass of SpikeBasis. The `basis` property is
    % derived from `basis_cs` using (for any k,c):
    %   basis(:,c,d) = { basis_cs(:,k,c)  for d = k + K*(c-1)
    %                  {        0         for any other d
    basis_cs
end


% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------


methods
    function obj = SpikeBasisCS(basis_cs, varargin)
        % SpikeBasisCS constructor
        %   obj = SpikeBasisCS(basis_cs, ...)
        %
        % Required arguments:
        %   basis_cs    [L x K x C] channel-specific spike basis waveforms
        % Optional parameters (key/value pairs) [default]:
        %   t0          Sample index (1..L) for t=0     [ 1 ]
        %   whitener    Whitener object                 [ none ]
        %   interp      Interpolator object             [ none ]
        basis_cs = gather(double(basis_cs));
        [L,K,C] = size(basis_cs);
        % Generate the [L x C x D] basis with the extra zeros
        basis = zeros(L,C,K,C);
        for c = 1:C
            basis(:,c,:,c) = reshape(basis_cs(:,:,c), [L 1 K]);
        end
        basis = reshape(basis, [L, C, K*C]);
        % Construct the object
        obj = obj@spkdec.SpikeBasis(basis, varargin{:});
        obj.basis_cs = basis_cs;
    end
    
    % Various forms of copy constructors
    basis_mod = copy_modifyCS(self, newbasis);
    basis_nonWh = copy_nonWh(self);
    
    % Convolution
    y = conv(self, x);
    x = convT(self, y);
end

methods (Static)
    % Construction from a non-CS object
    obj = from_basis(basis);
    
    % Unit tests
    test(varargin);
end

% ----------------------     Copy and serialization     ------------------------

methods
    function s = saveobj(self)
        s = saveobj@spkdec.SpikeBasis(self);
        s = rmfield(s,'basis');
        s.basis_cs = self.basis_cs;
    end
end
methods (Static)
    function obj = loadobj(s)
        s.whitener = spkdec.Whitener.loadobj(s.whitener);
        s.interp = spkdec.Interpolator.loadobj(s.interp);
        obj = spkdec.SpikeBasisCS(s.basis_cs, rmfield(s,'basis_cs'));
    end
end

% ------------------------------------------------------------------------------
% ===========================      Protected     ===============================
% ------------------------------------------------------------------------------

% Caches

properties (Access=protected)
    % ConvolverCS object, output of toConvCS() and used in convolution methods
    %
    % This has K*R kernels per channel, and convolver_cs.kernels_cs is
    % [L+W-1 x K*R x C].
    convolver_cs
end

methods (Access=protected)
    conv = toConvCS(self);
end

end
