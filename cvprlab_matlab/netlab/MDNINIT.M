function net = mdninit(net, prior, t, options)
%MDNINIT Initialise the weights in a Mixture Density Network.
%
%	Description
%
%	NET = MDNINIT(NET, PRIOR) takes a Mixture Density Network NET and
%	sets the weights and biases by sampling from a Gaussian distribution.
%	It calls MLPINIT for the MLP component of NET.
%
%	NET = MDNINIT(NET, PRIOR, T, OPTIONS) uses the target data T to
%	initialise the biases for the output units after initialising the
%	other weights as above.  It calls GMMINIT, with T and OPTIONS as
%	arguments, to obtain a model of the unconditional density of T.  The
%	biases are then set so that NET will output the values in the
%	Gaussian  mixture model.
%
%	See also
%	MDN, MLP, MLPINIT, GMMINIT
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

% Initialise network weights from prior: this gives noise around values
% determined later
net.mlp = mlpinit(net.mlp, prior);

if nargin > 2
  % Initialise priors, centres and variances from target data
  temp_mix = gmminit(net.mix, t, options);
  
  ncentres = net.mix.ncentres;
  dim_target = net.mix.nin;

  % Now set parameters in MLP to yield the right values.
  % This involves setting the biases correctly.
  
  % Priors
  net.mlp.b2(1:ncentres) = temp_mix.priors;
  
  % Centres
  end_centres = ncentres*(dim_target+1);
  net.mlp.b2(ncentres+1:end_centres) = ...
    reshape(temp_mix.centres, 1, ncentres*dim_target);
  
  % Variances
  net.mlp.b2((end_centres+1):net.mlp.nout) = ...
    log(temp_mix.covars);
end
