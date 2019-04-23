% Convolution with a set of channel-specific kernels
%
% Convolver properties (read-only):
% Dimensions
%   C           - Number of data channels
%   K           - Number of convolution kernels per channel
%   L           - Kernel length (#samples)
% Data
%   kernels     - [L x K x C] convolution kernels for each channel
%   wh_ch       - [C x C] cross-channel transform applied to convolution output
%   t0          - Sample index (1..L) corresponding to t=0
%
% Convolver methods:
%   Convolver   - Construct a new Convolver object
%   toMat       - Return a matrix representing the effective kernels
% Convolution
%   conv        - Perform the forward convolution
%   convT       - Perform the transpose convolution
% Object management
%   copy        - Create a copy of this handle object
%   saveobj     - Serialize a Convolver object to a struct
%   loadobj     - [Static] Deserialize a Convolver object from a given struct

classdef Convolver < matlab.mixin.Copyable


% ------------------------------------------------------------------------------
% ========================     Public properties     ===========================
% ------------------------------------------------------------------------------


properties (SetAccess=private, Dependent)
    % Number of data channels (convolution outputs)
    C
    
    % Number of convolution kernels per channel
    % The total number of kernels is K*C
    K
    
    % Kernel length (# of samples)
    % All kernels have the same length and <t0>
    L
end
methods
    function val = get.C(self), val = size(self.kernels,3); end
    function val = get.K(self), val = size(self.kernels,2); end
    function val = get.L(self), val = size(self.kernels,1); end
end

properties (SetAccess=protected)
    % [L x K x C] convolution kernels for each channel
    % Each of these is restricted to a single channel, as this amkes the
    % convolution faster to perform and is often convenient for interpreting the
    % resulting feature space. The forward convolution involves summing over the
    % K kernels for each channel.
    kernels
    
    % [C x C] cross-channel transform
    % This is applied at the end of the forward convolution (or start of the
    % transpose convolution).
    wh_ch = 1;
    
    % Kernel sample index (1..L) corresponding to t=0
    % This property is not actually used by any of the methods in this class,
    % but it is a useful piece of information to keep track of and it is
    % convenient for it to be associated with this object.
    t0 = 1;
end

properties (Constant, Access=private)
    errid_dim = 'spkdec:Convolver:DimMismatch';
    errid_arg = 'spkdec:Convolver:BadArg';
end


% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------


methods
    function obj = Convolver(kernels, varargin)
        % Convolver constructor
        %   obj = Convolver(kernels, ...)
        %
        % Required arguments:
        %   kernels   [L x K x C] convolution kernels
        % Optional parameters (key/value pairs) [default]:
        %   wh_ch     [C x C] cross-channel transform       [ eye(C) ]
        %   t0        Sample index (1..L) for t=0           [ 1 ]
        [~,~,C] = size(kernels);
        % Parse optional parameters
        ip = inputParser();
        ip.addParameter('wh_ch', eye(C), @ismatrix);
        ip.addParameter('t0', 1, @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        assert(isequal(size(prm.wh_ch),[C C]), obj.errid_arg, ...
            'wh_ch must be a square, [C x C] matrix');
        % Assign values
        obj.kernels = gather(double(kernels));
        obj.wh_ch = gather(double(prm.wh_ch));
        obj.t0 = gather(double(prm.t0));
    end
    
    
    % Convolution
    y = conv(self, x, varargin);
    x = convT(self, y, varargin);
    
    % Representation in a different form
    mat = toMat(self, varargin);
end

methods (Static)
    % Unit tests
    test(varargin);
end

% ----------------------     Copy and serialization     ------------------------

methods
    function s = saveobj(self)
        s = struct();
        for fn = {'kernels','wh_ch','t0'}
            s.(fn{1}) = self.(fn{1});
        end
    end
end
methods (Static)
    function obj = loadobj(s)
        obj = spkdec.Convolver(s.kernels, rmfield(s,'kernels'));
    end
end


% ------------------------------------------------------------------------------
% ===========================      Protected     ===============================
% ------------------------------------------------------------------------------


% Caches

properties (Access=protected)
    % [L x C x K x C] conv. kernels after the applying cross-channel transform.
    % This is used in toMat()
    kernels_full
    
    % [N_fft x K x C] kernels in frequency domain (as a causal filter).
    % This is used in get_kernels_hat()
    kernels_hat
end

methods (Access=protected)
    % Return the convolution kernels in frequency domain
    k_hat = get_kernels_hat(self, N);
end

% Convolution sub-steps

methods (Access=protected, Static)
    % Conversion to/from overlap-add or overlap-scrap batches
    y = vec_to_batch(x, N, ovlp, dupe);
    x = batch_to_vec(y, T, ovlp, add);
    % Convolution in frequency domain
    y_hat = conv_hat(x_hat, kern_hat);
    x_hat = convT_hat(y_hat, kern_hat);
end


end
