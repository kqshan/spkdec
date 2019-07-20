% Convolver subclass for the special case of channel-specific (CS) kernels
%
% ConvolverCS properties (read-only):
% Dimensions
%   L           - Kernel length (#samples)
%   C           - Number of data channels (convolution output dimension)
%   K           - Number of convolution kernels per channel (D = K*C)
%   D           - Number of convolution kernels (convolution input dimension)
% Data
%   kernels_cs  - [L x K x C] channel-specific kernels for each channel
%   kernels     - [L x C x D] convolution kernels
%   t0          - Sample index (1..L) corresponding to t=0
%   wh_ch       - [C x C] transform applied to channel-specific conv. output
%
% ConvolverCS methods:
%   ConvolverCS - Construct a new ConvolverCS object
% Convolution
%   conv        - Perform the forward convolution
%   convT       - Perform the transpose convolution
%   conv_batch  - Perform the forward convolution on short batches of data
% Object management
%   copy        - Create a copy of this handle object
%   saveobj     - Serialize a ConvolverCS object to a struct
%   loadobj     - [Static] Deserialize a ConvolverCS object from a given struct

classdef ConvolverCS < spkdec.Convolver


% ------------------------------------------------------------------------------
% ========================     Public properties     ===========================
% ------------------------------------------------------------------------------


properties (SetAccess=private, Dependent)
    % Number of convolution kernels per channel (D = K*C)
    K
end
methods
    function val = get.K(self), val = size(self.kernels_cs,2); end
end

properties (SetAccess=protected)
    % [L x K x C] convolution kernels for each channel
    %
    % These channel-specific kernels makes the convolution faster to perform and
    % is often convenient for interpreting the resulting feature space.
    %
    % `kernels_cs(:,k,c)` corresponds to `kernels(:,:,d)` by `d = k + K*(c-1)`
    % in terms of how the kernels are ordered, but note that the cross-channel
    % transform `wh_ch` is applied when deriving `kernels` from `kernels_cs'.
    kernels_cs
    % Why [L x K x C] rather than [L x C x K]? The latter would make the special
    % case of K==1 easier to deal with, and make SpikeBasisSC's subsample inter-
    % polation (in which the subsample-shifted kernels appear as additional
    % kernels) easier to implement. Alas, this is because this code evolved from
    % single-channel convolution and it's too much work to go back and change 
    % the order now.
    
    
    % [C x C] cross-channel transform applied to channel-specific conv. output
    %
    % This is applied at the end of the forward convolution (or start of the
    % transpose convolution). If this is not the identity, then the effective
    % kernels (the `kernels` property) will have multi-channel support even
    % though `kernels_cs` only specifies a single channel for each kernel.
    wh_ch = 1;
end


% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------


methods
    function obj = ConvolverCS(kernels_cs, varargin)
        % Convolver constructor
        %   obj = Convolver(kernels_cs, ...)
        %
        % Required arguments:
        %   kernels_cs  [L x K x C] channel-specific convolution kernels
        % Optional parameters (key/value pairs) [default]:
        %   wh_ch       [C x C] cross-channel transform       [ eye(C) ]
        %   t0          Sample index (1..L) for t=0           [ 1 ]
        [~,~,C] = size(kernels_cs);
        % Parse optional parameters
        ip = inputParser();
        ip.addParameter('wh_ch', eye(C), @ismatrix);
        ip.addParameter('t0', 1, @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        assert(isequal(size(prm.wh_ch),[C C]), spkdec.ConvolverCS.errid_arg, ...
            'wh_ch must be a square, [C x C] matrix');
        % Derive the kernel
        kernels = spkdec.ConvolverCS.compute_kernels(kernels_cs, prm.wh_ch);
        obj = obj@spkdec.Convolver(kernels, 't0',prm.t0);
        obj.kernels_cs = gather(double(kernels_cs));
        obj.wh_ch = gather(double(prm.wh_ch));
    end
end

methods (Static)
    % Unit tests
    test(varargin);
end

% ----------------------     Copy and serialization     ------------------------

methods
    function s = saveobj(self)
        s = struct();
        for fn = {'kernels_cs','wh_ch','t0'}
            s.(fn{1}) = self.(fn{1});
        end
    end
end
methods (Static)
    function obj = loadobj(s)
        obj = spkdec.ConvolverCS(s.kernels_cs, rmfield(s,'kernels_cs'));
    end
end


% ------------------------------------------------------------------------------
% ===========================      Protected     ===============================
% ------------------------------------------------------------------------------


% Caches

properties (Access=protected)
    % [N_fft x K x C] kernels in frequency domain (as a causal filter).
    % This is used in get_kernels_cs_hat()
    kernels_cs_hat
end

methods (Access=protected)
    % Return the convolution kernels in frequency domain
    kcs_hat = get_kernels_cs_hat(self, N);
    
    % Overload these to use a channel-specific implementation
    y_hat = conv_hat(self, x_hat);
    x_hat = convT_hat(self, y_hat);    
end

methods (Static, Access=protected)
    % Helper for constructing the full kernels
    kernels = compute_kernels(kernels_cs, wh_ch);
end

end
