function g = netgrad(w, net, x, t)
%NETGRAD Evaluate network error gradient for generic optimizers
%
%	Description
%
%	G = NETGRAD(W, NET, X, T) takes a weight vector W and a network data
%	structure NET, together with the matrix X of input vectors and the
%	matrix T of target vectors, and returns the gradient of the error
%	function evaluated at W.
%
%	See also
%	MLP, NETERR, NETOPT
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

unpakstr = [net.type, 'unpak'];
gradstr = [net.type, 'grad'];

net = feval(unpakstr, net, w);

g = feval(gradstr, net, x, t);
