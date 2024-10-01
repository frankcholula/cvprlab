function g = mdngrad(net, x, t)
%MDNGRAD Evaluate gradient of error function for Mixture Density Network.
%
%	Description
%	 G = MDNGRAD(NET, X, T) takes a mixture density network data
%	structure NET, a matrix X of input vectors and a matrix T of target
%	vectors, and evaluates the gradient G of the error function with
%	respect to the network weights. The error function is negative log
%	likelihood of the target data.  Each row of X corresponds to one
%	input vector and each row of T corresponds to one target vector.
%
%	See also
%	MDN, MDNFWD, MDNERR, MLPBKP
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

% Check arguments for consistency
errstring = consist(net, 'mdn', x, t);
if ~isempty(errstring)
  error(errstring);
end

[mixes, y, z] = mdnfwd(net, x);

% Compute gradients at MLP outputs: put the answer in deltas
ncentres = net.mix.ncentres;
dim_target = net.mix.nin;
nmixparams = net.mix.nparams;
ndata = size(x, 1);
deltas = zeros(ndata, net.mlp.nout);

e = ones(ncentres, 1);
f = ones(1, dim_target);

for i = 1:ndata
  post = gmmpost(mixes(i), t(i,:));

  % Calculate prior derivatives
  deltas(i, 1:ncentres) = mixes(i).priors - post;

  % Calculate centre derivatives
  centre_err = mixes(i).centres - e*t(i,:);
  deltas(i, (ncentres+1):(ncentres*(1+dim_target))) = ...
    reshape(((centre_err.*(post'*f))./(mixes(i).covars'*f)), ...
	1, ncentres*dim_target);
  
  % Calculate variance derivatives
  deltas(i, (ncentres*(1+dim_target)+1):nmixparams) = ...
    ((dim_target*e' - ((sum(centre_err.^2, 2))'./mixes(i).covars)).*post)./2;
end

% Now backpropagate the deltas through the MLP network
g = mlpbkp(net.mlp, x, z, deltas);