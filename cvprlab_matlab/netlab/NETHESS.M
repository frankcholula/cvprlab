function [h, varargout] = nethess(w, net, x, t, varargin)
%NETHESS Evaluate network Hessian
%
%	Description
%
%	H = NETHESS(W, NET, X, T) takes a weight vector W and a network data
%	structure NET, together with the matrix X of input vectors and the
%	matrix T of target vectors, and returns the value of the Hessian
%	evaluated at W.
%
%	[E, VARARGOUT] = NETHESS(W, NET, X, T, VARARGIN) also returns any
%	additional return values from the network Hessian function, and
%	passes additional arguments to that function.
%
%	See also
%	NETERR, NETGRAD, NETOPT
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

unpakstr = [net.type, 'unpak'];
hess_str = [net.type, 'hess'];

net = feval(unpakstr, net, w);

[s{1:nargout}] = feval(hess_str, net, x, t, varargin{:});
h = s{1};
for i = 2:nargout
  varargout{i-1} = s{i};
end
