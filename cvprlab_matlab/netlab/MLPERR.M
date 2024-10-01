function [e, edata, eprior] = mlperr(net, x, t)
%MLPERR	Evaluate error function for 2-layer network.
%
%	Description
%	E = MLPERR(NET, X, T) takes a network data structure NET together
%	with a matrix X of input vectors and a matrix T of target vectors,
%	and evaluates the error function E. The choice of error funcion
%	corresponds to the output unit activation function. Each row of X
%	corresponds to one input vector and each row of T corresponds to one
%	target vector.
%
%	[E, EDATA, EPRIOR] = MLPERR(NET, X, T) also returns the data and
%	prior components of the total error.
%
%	See also
%	MLP, MLPPAK, MLPUNPAK, MLPFWD, MLPBKP, MLPGRAD
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

% Check arguments for consistency
errstring = consist(net, 'mlp', x, t);
if ~isempty(errstring);
  error(errstring);
end
[y, z, a] = mlpfwd(net, x);

switch net.actfn

  case 'linear'        %Linear outputs

    edata = 0.5*sum(sum((y - t).^2));

  case 'logistic'      % Logistic outputs

    % Ensure that log(1-y) is computable: need exp(a) > eps
    maxcut = -log(eps);
    % Ensure that log(y) is computable
    mincut = -log(1/realmin - 1);
    a = min(a, maxcut);
    a = max(a, mincut);
    y = 1./(1 + exp(-a));
    edata = - sum(sum(t.*log(y) + (1 - t).*log(1 - y)));

  case 'softmax'       % Softmax outputs
  
    nout = size(a,2);
    % Ensure that sum(exp(a), 2) does not overflow
    maxcut = log(realmax) - log(nout);
    % Ensure that exp(a) > 0
    mincut = log(realmin);
    a = min(a, maxcut);
    a = max(a, mincut);
    temp = exp(a);
    y = temp./(sum(temp, 2)*ones(1,nout));
    % Ensure that log(y) is computable
    y(y<realmin) = realmin;
    edata = - sum(sum(t.*log(y)));

end

% Evaluate the data contribution to the error.
if isfield(net, 'beta')
  e1 = net.beta*edata;
else
  e1 = edata;
end

% Evaluate the prior contribution to the error.
if isfield(net, 'alpha')
  w = mlppak(net);
  if size(net.alpha) == [1 1]
    eprior = 0.5*(w*w');
    e2 = eprior*net.alpha;
  else
    eprior = 0.5*(w.^2)*net.index;
    e2 = eprior*net.alpha;
  end
else
  eprior = 0;
  e2 = 0;
end

e = e1 + e2;
