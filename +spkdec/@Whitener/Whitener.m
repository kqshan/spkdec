% Frequency- and cross-channel whitening operator
%
% Whitener properties (read-only):
% Dimensions
%   C           - Number of data channels
%   W           - Frequency-whitening filter length (# samples)
% Data
%   wh_filt     - [W x C] frequency-whitening filters
%   wh_ch       - [C x C] cross-channel whitening transform
%   delay       - Whitening filter delay (#samples)
%
% Whitener methods:
%   whiten      - Apply this whitener to the given data
%   toConv      - Return a Convolver object implementing this whitener
%   toMat       - Return a matrix implemetning this whitener
% Construction
%   Whitener    - Construct a new Whitener object
%   no_whiten   - [Static] Make a Whitener with no whitening
% Whitener generation helpers
%   makeWhFilt  - [Static] Make a set of frequency-whitening filters
%   makeWhCh    - [Static] Make a cross-channel whitening transform
% Object management
%   copy        - Create a copy of this handle object
%   saveobj     - Serialize a Whitener object to a struct
%   loadobj     - [Static] Deserialize a Whitener object from a given struct
%
% See also: spkdec.util.make_spk_whitener

classdef Whitener < matlab.mixin.Copyable


% ------------------------------------------------------------------------------
% ========================     Public properties     ===========================
% ------------------------------------------------------------------------------


properties (SetAccess=private, Dependent)
    % Number of data channels
    C
    
    % Length of the frequency-whitening filter (# samples)
    W
end
methods
    function x = get.C(self), x = size(self.wh_filt,2); end
    function x = get.W(self), x = size(self.wh_filt,1); end
end

properties (SetAccess=protected)
    % [W x C] frequency-whitening filters (one filter for each channel)
    % The filter delay of these filters is specified by the <delay> property
    wh_filt = 1
    
    % [C x C] cross-channel whitening transform
    % This is applied to each [C x 1] time sample independently, so if data_raw
    % is [C x T] then
    %   data_wh = wh_ch * data_raw
    % If the data were transposed (i.e. laid out as [T x C]), then
    %   data_wh_tr = (wh_ch * data_raw_tr.').'
    %              = data_raw_tr * wh_ch.'
    wh_ch = 1
    
    % Whitening filter delay (#samples)
    % Each of the C whitening filters are assumed to have the same delay.
    delay = 0
end

properties (Constant, Access=private)
    errid_arg = 'spkdec:Whitener:BadArg';
    errid_dim = 'spkdec:Whitener:BadDim';
end


% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------


% ---------------------------     Constructor     ------------------------------

methods
    function obj = Whitener(varargin)
        % Whitener constructor
        %   obj = Whitener(...)
        %
        % Optional parameters (key/value pairs) [default]:
        %   wh_filt     [W x C] whitening filter        [ ones(1,C) ]
        %   wh_ch       [C x C] cross-channel whitener  [ eye(C) ]
        %   delay       Whitening filter delay          [ (W-1)/2 ]
        ip = inputParser();
        ip.addParameter('wh_filt', []);
        ip.addParameter('wh_ch', []);
        ip.addParameter('delay', []);
        ip.parse( varargin{:} );
        prm = ip.Results;
        wh_filt = prm.wh_filt; wh_ch = prm.wh_ch; delay = prm.delay;
        % Decide on the channel count
        if ~isempty(wh_filt)
            C = size(wh_filt,2);
        elseif ~isempty(wh_ch)
            C = size(wh_ch,1); 
        else
            C = 1;
        end
        % Default values
        if isempty(wh_filt)
            wh_filt = ones(1,C);
        end
        if isempty(wh_ch)
            wh_ch = eye(C);
        end
        if isempty(delay)
            W = size(wh_filt,1);
            assert(mod(W,2)==1, obj.errid_arg, ['The default filter delay ' ...
                'assumes that the whitening filter length is odd']);
            delay = (W-1) / 2;
        end
        % Validation
        assert(size(wh_filt,2)==C, obj.errid_dim, 'wh_filt must be [W x C]');
        assert(ismatrix(wh_ch) && size(wh_ch,1)==size(wh_ch,2), ...
            obj.errid_arg, 'wh_ch must be a square matrix');
        assert(size(wh_ch,1)==C, obj.errid_dim, 'wh_ch must be [C x C]');
        assert(isscalar(delay), obj.errid_arg, 'delay must be a scalar');
        assert(mod(delay,1)==0, obj.errid_arg, 'delay must be an integer');
        % Update self
        obj.wh_filt = gather(double(wh_filt));
        obj.wh_ch = gather(double(wh_ch));
        obj.delay = gather(double(delay));
    end
end

% -----------------------     Other public methods     -------------------------

methods
    function conv = toConv(self)
        % Return a ConvolverCS object implementing this whitening operation
        %   conv = toConv(self)
        conv = spkdec.ConvolverCS( ...
            reshape(self.wh_filt, [self.W, 1, self.C]), ...
            'wh_ch',self.wh_ch, 't0',self.delay+1);
    end
    
    % Return a matrix implementing this whitening operation for finite support
    mat = toMat(self, L, varargin);
    
    % Perform the whitening operation
    y = whiten(self, x, varargin);
end

methods (Static)
    function obj = no_whiten(C)
        % Make a Whitener with no whitening
        %   obj = no_whiten(C)
        %
        % Required arguments:
        %   C       Number of channels
        obj = spkdec.Whitener('wh_filt',ones(1,C));
    end
    
    % Helpers for making whitening operators
    [wh_filt, wh_spec] = makeWhFilt(spect, varargin);
    wh_ch = makeWhCh(ch_cov, varargin);
    
    % Unit tests
    test(varargin);
end

% ----------------------     Copy and serialization     ------------------------

methods
    function s = saveobj(self)
        s = struct();
        for fn = {'wh_filt','wh_ch','delay'}
            s.(fn{1}) = self.(fn{1});
        end
    end
end
methods (Static)
    function obj = loadobj(s)
        obj = spkdec.Whitener(s);
    end
end


% ------------------------------------------------------------------------------
% ===========================      Protected     ===============================
% ------------------------------------------------------------------------------


% Caches of often-requested items
properties (Access=protected)
    % Convolver object to use for whitening
    convolver
    
    % [L+W-1 x C x L*C] whitening operation as a matrix (for a given support L)
    whitener_mat
end


end
