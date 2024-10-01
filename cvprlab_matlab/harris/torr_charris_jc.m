% %
% % HISTORY
%   2001 Philip Torr (philtorr@microsoft.com, phst@robots.ac.uk) at Microsoft
%   Created.
%   2002 John Collomosse (jpc@cs.bath.ac.uk) at Bath Univ.
%   Extensions to subpixel accuracy via quadratic surface fit
% 
%  Copyright © Microsoft Corp. 2002
%
%
% REF:	"A combined corner and edge detector", C.G. Harris and M.J. Stephens
%	Proc. Fourth Alvey Vision Conf., Manchester, pp 147-151, 1988.
%
%%to do: we might want to make this so it can either take a threshold or a fixed number of corners...
function [c_coord] = torr_charris_jc(im, ncorners, width, sigma, subpixel)

if (nargin < 2)
    error('not enough input in  charris');
elseif (nargin ==2)
    width = 3; %default
    sigma = 1;
end

if (nargin < 5)
    subpixel = 0;
end

mask = [-1 0 1; -1 0 1; -1 0 1] / 3;

% compute horizontal and vertical gradients
Ix = conv2(im, mask, 'valid');
Iy = conv2(im, mask', 'valid');

% compute squares amd product
Ixy = Ix .* Iy;
Ix = Ix.^2;
Iy = Iy.^2;

% smooth them
gmask = torr_gauss_mask(width, sigma);

gim = conv2(im, gmask, 'valid');


Ix = conv2(Ix, gmask, 'valid');
Iy = conv2(Iy, gmask, 'valid');
Ixy = conv2(Ixy, gmask, 'valid');

% computer cornerness
% c = (Ix + Iy) ./ (Ix .* Iy - Ixy.^2 + 1.0);
c = (Ix + Iy) - 0.04 * (Ix .* Iy - Ixy.^2);
%figure
%imagesc(c);
%figure
%c is smaller than before got border of 2 taken off all round
%size(c)

%compute max value around each pixel
%cmin = imorph(c, ones(3,3), 'min');
cmax = torr_max3x3(double(c));

% if pixel equals max, it is a local max, find index
ci3 = find(c == cmax);
cs3 = c(ci3);

[cs2,ci2] = sort(cs3); %ci2 2 is an index into ci3 which is an index into c


%put strongest 500 corners in a list cs together with indices ci

l = length(cs2)
cs = cs2(l-ncorners+1:l);
ci2s = ci2(l-ncorners+1:l);

ci = ci3(ci2s);

corn_thresh = cs(1);


%row and column of each corner

[nrows, ncols] = size(c);
%plus four for border
%   c_row = rem(ci,nrows) +4;
%   c_col = ( ci - c_row )/nrows + 1 +4;

c_row = rem(ci,nrows) +(width+1);
c_col = ( ci - c_row  +width+1)/nrows + 1 +width+1;

%   %to convert to x,y we need to convert from rows to y
c_coord = [c_col c_row];


if subpixel

    %   Extension to subpixel accuracy - John Collomosse 2002 jpc@cs.bath.ac.uk
    
    %   for each 'good corner' we must refine to subpixel accuracy, we fit
    %   a quadratic surface and find zero grad of that surface
           
    % 1) Fit the quadratic surface to a 3x3 neighbourhood centered around the 'corner' pixel
    %    (least squares fit)
    for lp=1:size(c_coord,1)
        x=c_coord(lp,1)-(width+1);    % convert into index into c (which is smaller than image)
        y=c_coord(lp,2)-(width+1);            
        A=[x^2      y^2       x*y           x     y     1   -c(y,x); ...
          (x-1)^2   (y-1)^2   (x-1)*(y-1)   x-1   y-1   1   -c(y-1,x-1); ...
          (x+1)^2   (y-1)^2   (x+1)*(y-1)   x+1   y-1   1   -c(y-1,x+1); ...
          (x-1)^2   (y+1)^2   (x-1)*(y+1)   x-1   y+1   1   -c(y+1,x-1); ...
          (x+1)^2   (y+1)^2   (x+1)*(y+1)   x+1   y+1   1   -c(y+1,x+1); ...
          x^2       (y+1)^2   x*(y+1)       x     y+1   1   -c(y+1,x); ...
          x^2       (y-1)^2   x*(y-1)       x     y-1   1   -c(y-1,x); ...
          (x-1)^2   y^2       (x-1)*y       (x-1) y   1     -c(y,x-1); ...
          (x+1)^2   y^2       (x+1)*y       (x+1) y   1     -c(y,x+1)];
        [u s v]=svd(A);
        res=v(:,end);
        res=res./res(end,1);

        % biquad. surface fitted, here are the params
        % I(x,y)=ax^2+by^2+cxy+dx+ey+f
        a=res(1);
        b=res(2);
        c2=res(3); % note c2, we used 'c' for something else
        d=res(4);
        e=res(5);
        f=res(6);
        
        coord=inv([2*a c2 ; c2 2*b])*[-d ; -e];        % solve to find the turning pt of surface
        
        % in some, rare, cases the max/min will be outside of the 3x3 block we fitted the
        % surface to. In this case, revert to single pixel accuracy.. (unclear to me why
        % this happens).
        if abs(x-coord(1))>1
            coord(1)=x;
        end
        if abs(y-coord(2))>1
            coord(2)=y;
        end
    
        coord=coord+width+1;  % corners converted back to image space
        c_coord(lp,1:2)=coord';
         
    end
    
end


