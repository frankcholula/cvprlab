function [h, dh] = mlphess(net, x, t, dh)
%MLPHESS Evaluate the Hessian matrix for a multi-layer perceptron network.
%
%	Description
%	H = MLPHESS(NET, X, T) takes an MLP network data structure NET, a
%	matrix X of input values, and a matrix T of target values and returns
%	the full Hessian matrix H corresponding to the second derivatives of
%	the negative log posterior distribution, evaluated for the current
%	weight and bias values as defined by NET.
%
%	[H, DH] = MLPHESS(NET, X, T) returns both the Hessian matrix H and
%	the contribution DH arising from the data dependent term in the
%	Hessian.
%
%	H = MLPHESS(NET, X, T, DH) takes a network data structure NET, a
%	matrix X of input values, and a matrix T of  target values, together
%	with the contribution DH arising from the data dependent term in the
%	Hessian, and returns the full Hessian matrix H corresponding to the
%	second derivatives of the negative log posterior distribution. This
%	version saves computation time if DH has already been evaluated for
%	the current weight and bias values.
%
%	See also
%	MLP, HESSCHEK, MLPHDOTV, EVIDENCE
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

% Check arguments for consistency
errstring = consist(net, 'mlp', x, t);
if ~isempty(errstring);
  error(errstring);
end

if nargin == 3
  % Data term in Hessian needs to be computed
  dh = datahess(net, x, t);
end

if isfield(net, 'beta')
  h = net.beta*dh;
else
  h = dh;
end

if isfield(net, 'alpha')
  if size(net.alpha) == [1 1]
    h = h + net.alpha*eye(net.nwts);
  else
    h = h + diag(net.index*net.alpha);
  end 
end

function dh = datahess(net, x, t)

dh = zeros(net.nwts, net.nwts);

for v = eye(net.nwts);
  dh(find(v),:) = mlphdotv(net, x, t, v);
end

return
