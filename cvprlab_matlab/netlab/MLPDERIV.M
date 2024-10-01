function g = mlpderiv(net, x)
%MLPDERIV Evaluate derivatives of network outputs with respect to weights.
%
%	Description
%	G = MLPDERIV(NET, X) takes a network data structure NET and a matrix
%	of input vectors X and returns a three-index matrix G whose I, J, K
%	element contains the derivative of network output K with respect to
%	weight or bias parameter J for input pattern I. The ordering of the
%	weight and bias parameters is defined by MLPUNPAK.
%
%	See also
%	MLP, MLPPAK, MLPGRAD, MLPBKP
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

% Check arguments for consistency
errstring = consist(net, 'mlp', x);
if ~isempty(errstring);
  error(errstring);
end

if ~strcmp(net.actfn, 'linear')
  error('Function not implemented except for linear outputs')
end

[y, z] = mlpfwd(net, x);

ndata = size(x, 1);

g = zeros(ndata, net.nwts, net.nout);
for k = 1 : net.nout
  delta = zeros(1, net.nout);
  delta(1, k) = 1;
  for n = 1 : ndata
    g(n, :, k) = mlpbkp(net, x(n, :), z(n, :), delta);
  end
end
