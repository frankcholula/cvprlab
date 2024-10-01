function pictout = pictNorm(pictin)

% pictout = pictNorm(pictin)
%   Normalise gray range in picture
% pre
%   pictin - a matrix of doubles represeting gray levels
% post
%   pictout - a matrix of doubles, between 0 and 1, representing
%             gray levels.
% See also
%	pictEvectNorm

vmin = min(min(pictin));
vrange = max(max(pictin)) - vmin;
pictout = (pictin-vmin)/vrange;

return;
