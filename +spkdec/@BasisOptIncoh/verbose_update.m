function verbose_update(self, iter, A, X)
% Update the verbose output
%   verbose_update(self, iter, A, X)
%
% Required arguments:
%   iter    Current iteration #
%   A       Spike basis
%   X       Detected spikes in feature space
%
% This accesses self.lip, so be careful about when you update that relative to
% calling this method.

% Exit early if verbose output is disabled
verbose = self.verbose;
if ~verbose, return; end

% See if we need to do an update
is_first_iter = (iter == 0);
is_final_iter = (iter == self.n_iter);
needs_iter_update = (verbose > 1) && (is_first_iter || is_final_iter || ...
    (iter >= self.vb_last + self.vb_period));

% Compute the error if desired
if is_first_iter || is_final_iter || needs_iter_update
    err = self.eval_error(A, X);
    err = sum(err.^2,'all');
    coh = sum((self.coh_L'*A).^2,'all');
    obj = err + self.coh_penalty*coh;
end
% Store this if it's the first iter
if is_first_iter
    self.err_0 = obj;
end

% Provide the iteration-level update if desired
if needs_iter_update
    % Compute some other things
    change = sum((A-self.A0).^2,'all');
    rel_err = err / self.norm_y;
    rel_change = change / sum(self.A0.^2,'all');
    rel_coh = coh / self.norm_y;
    rel_obj = (obj + self.lambda*change) / self.norm_y;
    % Print
    fprintf(self.vb_format, iter, toc(self.t_start), ...
        self.lip_max/self.lip, rel_err, rel_change, rel_coh, rel_obj);
    self.vb_last = iter;
end

% Provide the final update if desired
if is_final_iter
    fprintf('  Completed in %.1f sec with an improvement of %.4f\n', ...
        toc(self.t_start), (self.err_0 - obj)/self.norm_y);
end

end
