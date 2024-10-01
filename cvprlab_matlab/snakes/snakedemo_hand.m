%%%%%%%% CATMULL-ROM Splines as Active Contours (snakes)  -  a demo 

close all;
clear all;

% Parameters - test data synthesis
BLOBSIZE=5;     % size of test blob
JAGGEDNESS=9;   % spikyness of test blob
DECAY=0.5;      % rate at which field strength decays as move away from edge

% Parameters - snake
SNAKECP=90;     % number of control points on snake - MUST BE EVEN!


%%% Most relevant are ALPHA, BETA, GAMMA which weight the energy term for
%%% the snake (1st deriv, 2nd deriv, external field, respectively)

% Set this to ZERO if you have the image processing toolbox installed
% and are therefore able to use the bwdist function
USE_PRECOMPUTED_FIELD=1;

%%%%%%%%%%%% 1) Generate field for the snake
if (USE_PRECOMPUTED_FIELD) 
    field=double(imread('snakefield_real.bmp'))./255;    
else
    img=double(imread('bluehand.jpg'))./255;
    mask=double(img(:,:,1)<0.5); % use the red channel to make a simple mask of hand

    K=[1 2 1 ; 0 0 0 ; -1 -2 -1];   % Sobel kernel for dI/dy
    dx=conv2(mask,K,'same')./4;
    dy=conv2(mask,K','same')./4;
    mag=sqrt(dx.^2 + dy.^2);
    mask=mag>0.5;

    field=bwdist(mask);
    field=double(1-pictNorm(field).^DECAY);  % the snake code assumes energy field in range [0,1]
end

imgshow(field);
hold on;

%%%%%%%%%%% 2) Generate initial position of contour (in a big circle)
t=0:(2*pi)/(SNAKECP-1):2*pi;
RADIUS=190;
INITCONTOUR=(size(field,2)/2)+[cos(t);sin(t)]*RADIUS;
plot([INITCONTOUR(1,:) INITCONTOUR(1,1)],[INITCONTOUR(2,:) INITCONTOUR(2,1)],'r');

%%%%%%%%%%% 3) Perform snake optimization
FITTEDCONTOUR=snkfit(field,INITCONTOUR,0);
plot([FITTEDCONTOUR(1,:) FITTEDCONTOUR(1,1)],[FITTEDCONTOUR(2,:) FITTEDCONTOUR(2,1)],'g');
