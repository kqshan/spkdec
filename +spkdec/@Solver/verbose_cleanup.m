function verbose_cleanup(self, n_iter, spk, resid)
% Conclude the verbose output
% 
% Required arguments:
%   n_iter      Total number of iterations
%   spk         Final detected spikes (Spikes object)
%   resid       [T+V x C] final residual (b - A*x)

% Print the summary
if self.verbose
    % Compute the values
    if (self.verbose > 1)
        b_norm = norm(self.b,'fro')^2;
        err = self.last_err;
        nnz = self.last_nnz;
    else
        b_norm = self.last_err;
        err = norm(resid,'fro').^2;
        nnz = spk.N;
    end
    [T, ~, ~, C] = size(self.At_b);
    % Print
    fprintf('  Completed in %d iter at %s (%.1f sec elapsed)\n', ...
        n_iter, datestr(now(),31), toc(self.t_start));
    fprintf('  Detected %d spikes (%0.6f *T)\n', nnz, nnz/T);
    fprintf('  Residual %.3g rms (%0.4f *||b||^2)\n', sqrt(err/T/C),err/b_norm);
end

% Cleanup the object-level cache
self.vb_format = []; self.last_err = []; self.last_nnz = [];

end
