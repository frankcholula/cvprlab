function [y, z, a] = mlpfwd(net, x)
%MLPFWD	Forward propagation through 2-layer network.
%
%	Description
%	Y = MLPFWD(NET, X) takes a network data structure NET together with a
%	matrix X of input vectors, and forward propagates the inputs through
%	the network to generate a matrix Y of output vectors. Each row of X
%	corresponds to one input vector and each row of Y corresponds to one
%	output vector.
%
%	[Y, Z] = MLPFWD(NET, X) also generates a matrix Z of the hidden unit
%	activations where each row corresponds to one pattern.
%
%	[Y, Z, A] = MLPFWD(NET, X) also returns a matrix A  giving the summed
%	inputs to each output unit, where each row corresponds to one
%	pattern.
%
%	See also
%	MLP, MLPPAK, MLPUNPAK, MLPERR, MLPBKP, MLPGRAD
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

% Check arguments for consistency
errstring = consist(net, 'mlp', x);
if ~isempty(errstring);
  error(errstring);
end

ndata = size(x, 1);

z = tanh(x*net.w1 + ones(ndata, 1)*net.b1);
a = z*net.w2 + ones(ndata, 1)*net.b2;

switch net.actfn

  case 'linear'        %Linear outputs

    y = a;

  case 'logistic'    % Logistic outputs

    y = 1./(1 + exp(-a));

  case 'softmax'   % Softmax outputs
  
    temp = exp(a);
    nout = size(a,2);
    y = temp./(sum(temp,2)*ones(1,nout));

end
