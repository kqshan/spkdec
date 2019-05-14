% Sub-sample interpolator for waveforms of finite support
%
% Interpolator properties (read-only):
% Dimensions
%   L           - Waveform length
%   R           - Interpolation ratio (1 = no interpolation)
% Data
%   shifts      - [L x L x R] sub-sample shift matrices
%
% Interpolator methods:
%   interp      - Interpolate a [L x 1] vector to an [R*L x 1] vector
%   shift       - Apply the selected shift to the given [L x 1] vector
%   shiftArr    - Apply the selected shifts to an [L x C x N] data array
% Construction
%   Interpolator - Construct a new Interpolator object, given shift matrices
%   make_interp  - [Static] Make an Interpolator using a variety of methods
%   no_interp    - [Static] Make an Interpolator with no interpolation (R=1)
% Object management
%   copy        - Create a copy of this handle object
%   saveobj     - Serialize an Interpolator object to a struct
%   loadobj     - [Static] Deserialize a Interpolator object from a given struct
%
% Sub-sample interpolation is used to apply small temporal shifts (less than one
% sampling period, hence "sub-sample") to a given waveform.
%
% During the spike detection process, this provides temporal super-resolution on
% the detected spike times. This is not super useful on its own (unelss you need
% 10-microsecond resolution on your spike times), but it also makes the detected
% spike features invariant to the relative timing of the spike vs. acquisition
% sample clock. This is pretty useful, since temporal shifts (due to imperfect
% spike centering/alginment) can otherwise account for a substantial fraction
% (like 30%) of the observed spike variability.
%
% During spike basis optimization (a.k.a. learning the spike basis waveforms),
% the ability to perform sub-sample spike alignment using this Interpolator's
% shift operators reduces the incentive to represent such shifts using the basis
% itself. Without this, the 2nd component of the spike basis often ends up
% looking like the derivative of the 1st.

classdef Interpolator < matlab.mixin.Copyable

% ------------------------------------------------------------------------------
% ========================     Public properties     ===========================
% ------------------------------------------------------------------------------


properties (SetAccess=private, Dependent)
    % Waveform length
    % This interpolator is restricted to operating on waveforms with a given
    % support. This may seem like a strange restriction, but it is what allows
    % us to express the sub-sample shift operation as an explicit matrix.
    L
    
    % Interpolation ratio (1 = no interpolation)
    % R must be an integer >= 1.
    R
end
methods
    function val = get.L(self), val = size(self.shifts,1); end
    function val = get.R(self), val = size(self.shifts,3); end
end

properties (SetAccess=protected)
    % [L x L x R] sub-sample shift matrices
    % Each of these [L x L] matrices explicitly defines how to shift a vector:
    %   shifted_vector = shifts(:,:,r) * input_vector
    % If this object was created using Interpolator.make_interp(), then
    % shifts(:,:,1) will be an identity matrix, and the subsequent matrices
    % shift the waveform backwards in time by fractions of a sample.
    shifts
end

properties (Constant, Access=private)
    errid_arg = 'spkdec:Interpolator:BadArg';
    errid_dim = 'spkdec:Interpolator:DimMismatch';
end


% ------------------------------------------------------------------------------
% ========================      Public methods      ============================
% ------------------------------------------------------------------------------


methods
    function obj = Interpolator(shifts)
        % Interpolator constructor
        %   obj = Interpolator(shifts)
        %
        % Required arguments:
        %   shifts      [L x L x R] array of sub-sample shift matrices
        %               or [L*R x L] interpolation array
        [L_,L,R] = size(shifts);
        if (L ~= L_)
            % Deal with the case where shifts was given as [R*L x L]
            assert(R==1, obj.errid_arg, ['shifts either must be an ' ...
                '[L x L x R] or an [R*L x L] matrix']);
            R = L_/L;
            assert(mod(R,1)==0, obj.errid_arg, ['If shifts is an ' ...
                '[R*L x L] matrix, then R must be an integer']);
            shifts = reshape(shifts, [R L L]);
            shifts = flipud(shifts); % shifts must be forward shifts
            shifts = permute(shifts, [2 3 1]);
        end
        obj.shifts = gather(double(shifts(:,:,:)));
    end
    
    % Main interpolation methods
    y = interp(self, x);
    y = shift(self, x, r, trans);
    Y = shiftArr(self, X, r, trans);
end

methods (Static)
    % Helper for constructing shift matrices
    [obj, interpMat] = make_interp(L, R, varargin);
    
    % Explicit helper for no interpolation
    function obj = no_interp(L)
        % Create an Interpolator object with no interpolation (R=1)
        %   obj = no_interp(L)
        %
        % Returns:
        %   obj     Interpolator object
        % Required arguments:
        %   L       Signal length (#samples)
        obj = spkdec.Interpolator(eye(L));
    end
    
    % Unit tests
    test(varargin);
end

% ----------------------     Copy and serialization     ------------------------

methods
    function s = saveobj(self)
        s = struct();
        s.shifts = self.shifts;
    end
end
methods (Static)
    function obj = loadobj(s)
        obj = spkdec.Interpolator(s.shifts);
    end
end

end
