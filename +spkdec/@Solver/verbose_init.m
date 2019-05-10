function verbose_init(self)
% Initialize the verbose output
%   verbose_init(self)

% Summary output
if self.verbose
    % We start with x = 0 and therefore resid = b
    err = norm(self.b,'fro')^2;
    self.last_err = gather(double(err));
    self.last_nnz = 0;
    % Print the header
    [T, K, R, C] = size(self.At_b);
    fprintf('solve() started at %s\n', datestr(now(),31));
    fprintf('  T=%d, C=%d, K=%d, R=%d, data %.3g rms\n', ...
        T, C, K, R, sqrt(err/T/C));
end

% Set up the iteration-level updates
if (self.verbose > 1)
    % Set up the header and display format
    hd1 = '              |';
    hd2 = '  iter t(sec) |';
    fmt = '  %4d %6.1f |';
    hd1 = [hd1 '       normalized by T        |       delta      |\n'];
    hd2 = [hd2 '      phi   rec_err       nnz |     phi      nnz |\n'];
    fmt = [fmt '%9.4g %9.4g %9.6f |%8.3g %8.6f |\n'];
    fprintf(hd1); fprintf(hd2);
    self.vb_format = fmt;
end

end
