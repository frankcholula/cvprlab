function [g, gdata, gprior] = mlpgrad(net, x, t)
%MLPGRAD Evaluate gradient of error function for 2-layer network.
%
%	Description
%	G = MLPGRAD(NET, X, T) takes a network data structure NET  together
%	with a matrix X of input vectors and a matrix T of target vectors,
%	and evaluates the gradient G of the error function with respect to
%	the network weights. The error funcion corresponds to the choice of
%	output unit activation function. Each row of X corresponds to one
%	input vector and each row of T corresponds to one target vector.
%
%	[G, GDATA, GPRIOR] = MLPGRAD(NET, X, T) also returns separately  the
%	data and prior contributions to the gradient. In the case of multiple
%	groups in the prior, GPRIOR is a matrix with a row for each group and
%	a column for each weight parameter.
%
%	See also
%	MLP, MLPPAK, MLPUNPAK, MLPFWD, MLPERR, MLPBKP
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

% Check arguments for consistency
errstring = consist(net, 'mlp', x, t);
if ~isempty(errstring);
  error(errstring);
end
[y, z] = mlpfwd(net, x);
delout = y - t;

gdata = mlpbkp(net, x, z, delout);

% Evaluate the data contribution to the gradient.
if isfield(net, 'beta')
  g1 = gdata*net.beta;
else
  g1 = gdata;
end

% Evaluate the prior contribution to the gradient.
if isfield(net, 'alpha')
  w = mlppak(net);
  if size(net.alpha) == [1 1]
    gprior = w;
    g2 = net.alpha*gprior;
  else
    ngroups = size(net.alpha, 1);
    gprior = net.index'.*(ones(ngroups, 1)*w);
    g2 = net.alpha'*gprior;
  end
else
  gprior = 0;
  g2 = 0;
end

g = g1 + g2;
