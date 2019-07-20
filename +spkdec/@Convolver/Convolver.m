% Causal convolution with a set of kernels
%
% Convolver properties (read-only):
% Dimensions
%   L           - Kernel length (#samples)
%   C           - Number of data channels (convolution output dimension)
%   D           - Number of convolution kernels (convolution input dimension)
% Data
%   kernels     - [L x C x D] convolution kernels
%   t0          - Sample index (1..L) corresponding to t=0
%
% Convolver methods:
%   Convolver   - Construct a new Convolver object
% Convolution
%   conv        - Perform the forward convolution
%   convT       - Perform the transpose convolution
%   conv_batch  - Perform the forward convolution on short batches of data
% Object management
%   copy        - Create a copy of this handle object
%   saveobj     - Serialize a Convolver object to a struct
%   loadobj     - [Static] Deserialize a Convolver object from a given struct

classdef Convolver < matlab.mixin.Copyable


% ------------------------------------------------------------------------------
% ========================     Public properties     ===========================
% ------------------------------------------------------------------------------


properties (SetAccess=private, Dependent)
    % Kernel length (# of samples)
    %
    % All kernels have the same length and <t0>
    L

    % Number of data channels (convolution output dimension)
    C
    
    % Number of convolution kernels (convolution input dimension)
    D
end
methods
    function val = get.L(self), val = size(self.kernels,1); end
    function val = get.C(self), val = size(self.kernels,2); end
    function val = get.D(self), val = size(self.kernels,3); end
end

properties (SetAccess=protected)
    % [L x C x D] convolution kernels
    %
    % Each of the D kernels is represented as [L x C] waveforms on each channel.
    % If all kernels are restricted to a single channel each, and there is the
    % same number of kernels for each channel, then consider using the channel-
    % specific subclass spkdec.ConvolverCS instead.
    kernels
    
    % Kernel sample index (1..L) corresponding to t=0
    %
    % This property is not actually used by any of the methods in this class,
    % but it is a useful piece of information to keep track of and it is
    % convenient for it to be associated with this object.
    t0 = 1;
end

properties (Constant, Hidden)
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
        %   kernels   [L x C x D] convolution kernels
        % Optional parameters (key/value pairs) [default]:
        %   t0        Sample index (1..L) for t=0           [ 1 ]
        ip = inputParser();
        ip.addParameter('t0', 1, @isscalar);
        ip.parse( varargin{:} );
        prm = ip.Results;
        % Assign values
        obj.kernels = gather(double(kernels));
        obj.t0 = gather(double(prm.t0));
    end
    
    
    % Convolution
    y = conv(self, x, varargin);
    x = convT(self, y, varargin);
    y = conv_batch(self, x);
end

methods (Static)
    % Unit tests
    test(varargin);
end

% ----------------------     Copy and serialization     ------------------------

methods
    function s = saveobj(self)
        s = struct();
        for fn = {'kernels','t0'}
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
    % [N_fft x C x D] kernels in frequency domain (as a causal filter).
    % This is used in get_kernels_hat()
    kernels_hat
end

methods (Access=protected)
    % Return the convolution kernels in frequency domain
    k_hat = get_kernels_hat(self, N);
    
    % Convolution in frequency domain
    y_hat = conv_hat(self, x_hat);
    x_hat = convT_hat(self, y_hat);
end

% Convolution sub-steps

methods (Access=protected, Static)
    % Conversion to/from overlap-add or overlap-scrap batches
    y = vec_to_batch(x, N, ovlp, dupe);
    x = batch_to_vec(y, T, ovlp, add);
end


end
