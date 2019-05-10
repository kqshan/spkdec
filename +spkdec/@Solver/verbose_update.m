function verbose_update(self, iter, spk, resid)
% Update the verbose output for the current iteration
%   verbose_update(self, iter, spk, resid)
%
% Required arguments:
%   iter      Current iteration number
%   spk       Detected spikes (Spikes object)
%   resid     [T+V x C] residual (b - A*spk)

if (self.verbose > 1)
    % Compute the stats
    nnz = spk.N;
    err = norm(resid,'fro')^2;
    err = double(gather(err));
    phi = err + self.beta * nnz;
    last_phi = self.last_err + self.beta * self.last_nnz;
    % Print
    T = size(self.At_b,1);
    fprintf(self.vb_format, iter, toc(self.t_start), ...
        phi/T, err/T, nnz/T, (last_phi-phi)/T, (nnz-self.last_nnz)/T);
    % Update
    self.last_err = err;
    self.last_nnz = nnz;
end

end
