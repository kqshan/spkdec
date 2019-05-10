function delta = eval_improvement(self, resid)
% Evaluate the improvement in reconstruction error from adding a spike
%   delta = eval_improvement(self, resid)
%
% Returns:
%   delta     [R x T] change in ||A*x-b||^2 from adding a spike at each (r,t)
% Required arguments:
%   resid     [T+V x C] spike residual (whitened)

% Perform the transpose convolution
A = self.A;
convT_resid = A.convT(resid);

% Compute the improvement in reconstruction error
delta = A.getDelta(convT_resid);

end
