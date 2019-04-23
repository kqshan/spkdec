function wh_ch = makeWhCh(ch_cov, varargin)
% Construct a cross-channel whitener for the given cross-channel covariance
%   wh_ch = Whitener.makeWhCh(ch_cov, ...)
%
% Returns:
%   wh_ch       [C x C] cross-channel whitening transform
% Required arguments:
%   ch_cov      [C x C] cross-channel covariance
% Optional parameters (key/value pairs) [default]:
%   method      Whitener construction method: {eig, chol, sqrtm}    ['sqrtm']
%   reduce_err  Reduce the error wh_ch*ch_cov*wh_ch' - eye(C)       [ true ]
%
% This constructs a [C x C] matrix wh_ch so that
%   wh_ch * ch_cov * wh_ch' == eye(C)
% There are a couple of different ways to do this, and the different methods
% achieve additional properties:
%   eig     wh_ch*wh_ch' is diagonal
%   chol    wh_ch is lower triangular
%   sqrtm   wh_ch is symmetric
%
% If reduce_err==true, then we take some constrained gradient descent steps to
% reduce the error ||wh_ch*ch_cov*wh_ch' - eye(C)||.

% Input validation
[C,C_] = size(ch_cov);
assert(C==C_, spkdec.Whitener.errid_arg, 'ch_cov must be a square matrix');

% Optional parameters
ip = inputParser();
ip.addParameter('method', 'eig', @ischar);
ip.addParameter('reduce_err', true, @isscalar);
ip.parse( varargin{:} );
prm = ip.Results;

% Construct the whitener
method = prm.method;
switch (method)
    case 'eig'
        [V,d] = eig(ch_cov,'vector');
        wh_ch = diag(1./sqrt(d)) * V';
    case 'chol'
        wh_ch = inv(chol(ch_cov,'lower'));
    case 'sqrtm'
        wh_ch = inv(sqrtm(ch_cov));
    otherwise
        error(spkdec.Whitener.errid_arg, 'Unsupported method "%s"',method);
end

% Perform gradient descent to reduce the error
% By the way, this is completely unnecessary and there is no reason why you
% would need this level of precision for this application.
if prm.reduce_err
    % Decide on the number of steps
    n_steps = ceil(2*sqrt(C));
    if strcmp(method,'chol')
        n_steps = ceil(n_steps / 4);
    end
    n_steps = max(3, n_steps);
    
    % Perform the iterations
    err_log = zeros(n_steps+1,1); % For debugging
    for ii = 1:n_steps
        % Compute the gradient
        wh_err = wh_ch * ch_cov * wh_ch' - eye(C);
        err_log(ii) = norm(wh_err,'fro');
        grad = wh_err * wh_ch * ch_cov;
        % Project this into the feasible space
        switch (method)
            case 'eig'
                % Do nothing; it was never exactly orthogonal anyways
            case 'chol'
                grad = grad .* (wh_ch ~= 0);
            case 'sqrtm'
                grad = (grad + grad.') / 2;
        end
        % Take the gradient descent step
        step_size = 0.5 / norm(wh_ch * ch_cov)^2;
        wh_ch = wh_ch - step_size * grad;
    end
    err_log(end) = norm(wh_ch * ch_cov * wh_ch' - eye(C),'fro'); %#ok<NASGU>
    % disp(err_log/eps())
end

end
