% Detected spike times and features
%
% Spikes properties:
% Dimensions
%   N           - Number of spikes
% Data
%   t           - [N x 1] spike times (integer)
%   r           - [N x 1] spike sub-sample shift index (1..R)
%   X           - [D x N] spike features (may be empty)
%
% Spikes methods:
% Construction and modification
%   Spikes      - Constructor
%   setFeat     - Set the features
%   shiftTimes  - Shift the spike times by a given offset
%   addSpikes   - Append new spikes to this collection
% Data conversion
%   toFull      - Return a [T x D x R] dense representation of the spikes
%   subset      - Return a subset of these spikes
%   concat      - Concatenate a set of Spikes objects
% Object management
%   copy        - Make a copy of this handle object
%   saveobj     - Serialize a Spikes object to struct
%   loadobj     - [Static] Deserialize a Spikes object from a struct

classdef Spikes < matlab.mixin.Copyable

% ------------------------------------------------------------------------------
% ========================     Public properties     ===========================
% ------------------------------------------------------------------------------


properties (SetAccess=private, Dependent)
    % Number of spikes
    N
end
methods
    function val = get.N(self), val = length(self.t); end
end

properties (SetAccess=protected)
    % [N x 1] spike times (integer)
    t
    
    % [N x 1] sub-sample shift index (1..R)
    r
    
    % [D x N] spike features (may be empty)
    X
end

properties (Access=private)
    errid_dim = 'spkdec:Spikes:DimMismatch';
    errid_arg = 'spkdec:Spikes:BadArg';
end


% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------


methods
    function obj = Spikes(t, r, X)
        % Spikes constructor
        %   obj = Spikes(t, r, X)
        %
        % Optional arguments [default]:
        %   t       [N x 1] spike times (integer)           [ none ]
        %   r       [N x 1] sub-sample shift index (1..R)   [ ones(N,1) ]
        %   X       [D x N] spike features                  [ none ]
        if nargin < 1, t = zeros(0,1); else, t = t(:); end
        assert(all(mod(t,1)==0), obj.errid_arg, ...
            't must be a vector of integers');
        N = numel(t);
        if nargin < 2, r = ones(N,1); else, r = r(:); end
        assert(numel(r)==N, obj.errid_dim, 't and r must be the same length');
        assert(all(r >= 1) && all(mod(r,1)==0), obj.errid_arg, ...
            'r must be a vector of positive integers');
        if nargin < 3, X = []; else, X = X(:,:); end
        assert(isempty(X) || size(X,2)==N, obj.errid_dim, ...
            'X must either be empty or a [D x N] matrix with N==%d',N);
        % Assign to object
        obj.t = t; obj.r = r; obj.X = X;
    end
end

methods
    % Post-construction data modification
    
    function setFeat(self, X)
        % Set the features of this Spikes object
        %   setFeat(self, X)
        %
        % Required arguments:
        %   X       [D x N] spike features
        [~,N_] = size(X);
        assert(ismatrix(X) && N_==self.N, self.errid_dim, ...
            'X must be a [D x N] matrix with N==%d',self.N);
        self.X = X;
    end
    
    function shiftTimes(self, delta_t)
        % Shift the spike times by a given offset
        %   shiftTimes(self, delta_t)
        %
        % Required arguments:
        %   delta_t     [N x 1] or scalar time shift (#samples)
        %
        % This sets self.t = self.t + delta_t.
        assert(all(mod(delta_t,1)==0),self.errid_arg,'delta_t must be integer');
        self.t = self.t + delta_t(:);
    end
    
    addSpikes(self, new_spk);
    
    
    % Conversion methods that do not modify the object itself
    
    spikes = toFull(self, varargin);
    y = subset(self, mask);
    spk_all = concat(self, varargin);
end


% ----------------------     Copy and serialization     ------------------------

methods
    function s = saveobj(self)
        s = struct();
        for fn = {'t','r','X'}, s.(fn{1}) = self.(fn{1}); end
    end
end
methods (Static)
    function obj = loadobj(s)
        obj = spkdec.Spikes(s.t, s.r, s.X);
    end
end

end
