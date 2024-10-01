function [mixes, y, z, a] = mdnfwd(net, x)
%MDNFWD	Forward propagation through Mixture Density Network.
%
%	Description
%	 MIXES = MDNFWD(NET, X) takes a mixture density network data
%	structure NET and a matrix X of input vectors, and forward propagates
%	the inputs through the network to generate an array MIXES of output
%	mixture models.  Each row of X represents one input vector and the
%	corresponding row of MIXES represents the data structure vector of
%	the corresponding mixture model for the conditional probability of
%	target vectors.
%
%	[MIXES, Y, Z] = MDNFWD(NET, X) also generates a matrix Y of the
%	outputs of the MLP and a matrix Z of the hidden unit activations
%	where each row corresponds to one pattern.
%
%	[MIXES, Y, Z, A] = MLPFWD(NET, X) also returns a matrix A  giving the
%	summed inputs to each output unit, where each row  corresponds to one
%	pattern.
%
%	See also
%	GMM, MDN, MDNERR, MDNGRAD, MLPFWD
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

% Check arguments for consistency
errstring = consist(net, 'mdn', x);
if ~isempty(errstring)
  error(errstring);
end

% Extract mlp and mixture model descriptors
mlpnet = net.mlp;
mix = net.mix;

ncentres = mix.ncentres;	% Number of components in mixture model
dim_target = mix.nin;		% Dimension of targets
nparams = mix.nparams;		% Number of parameters in mixture model

% Propagate forwards through MLP
[y, z, a] = mlpfwd(mlpnet, x);

% Convert output values into mixture model parameters

% Use softmax to calculate priors
% Prevent overflow and underflow: use same bounds as glmfwd
% Ensure that sum(exp(y), 2) does not overflow
maxcut = log(realmax) - log(ncentres);
% Ensure that exp(y) > 0
mincut = log(realmin);
temp = min(y(:,1:ncentres), maxcut);
temp = max(temp, mincut);
temp = exp(temp);
mixpriors = temp./(sum(temp, 2)*ones(1,ncentres));

% Centres are just copies of network outputs

% Variances are exp of network outputs
mixwidths = exp(y(:,(ncentres*(1+dim_target)+1):nparams));

% Now build up all the mixture model weight vectors
ndata = size(x, 1);

% Construct ndata structures containing the mixture model information.
% First allocate the memory.
f = fieldnames(net.mix);
mixes = cell(size(f, 1), 1, ndata);
mixes = cell2struct(mixes, f,1);

% Then fill each structure in turn using gmmunpak.  Assume that spherical
% covariance structure is used.
for i = 1:ndata
  mixes(i) = gmmunpak(mix, [mixpriors(i,:), ...
      reshape(y(i, (ncentres+1):ncentres*(1+dim_target)), ...
      1, ncentres*dim_target), ...
      mixwidths(i,:)]);
end
