function verbose_init(self)
% Initialize the verbose output of this optimizer
%   verbose_init(self)
%
% Call this after defining self.Y

verbose = self.verbose;

% Print a simple header
if verbose
    [~,N,S] = size(self.Y); [L,K,C] = size(self.A0);
    fprintf('optimize() started at %s (L=%d, K=%d, C=%d, N=%d)\n', ...
        datestr(now(),31), L, K, C, N);
    s_idx = (S-1)/2 + 1; % Index corresponding to zero shift
    norm_y = norm(self.Y(:,:,s_idx),'fro')^2;
    norm_y = gather(double(norm_y));
else
    norm_y = NaN;
end

% Determine if we need iteration-level updates
if verbose > 1
    % Set up the header and display format
    hdr = '  iter      t |  step  Rec.Err   Change  Combined |\n';
    fmt = '  %4d %6.1f | %5.1f  %7.5f  %7.5f  %8.5f |\n';
    fprintf(hdr);
    % Decide on the update period
    update_period = round(1/(verbose-1));
else
    fmt = '';
    update_period = Inf;
end

% Set the class properties
self.vb_format = fmt;
self.vb_period = update_period;
self.vb_last = 0;
self.norm_y = norm_y;

end
