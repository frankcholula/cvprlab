%%%%%%%% CATMULL-ROM Splines as Active Contours (snakes)  -  a demo 

close all;
clear all;

% Parameters - test data synthesis
BLOBSIZE=5;     % size of test blob
JAGGEDNESS=9;   % spikyness of test blob
DECAY=0.5;      % rate at which field strength decays as move away from edge


% Parameters - snake
SNAKECP=14;     % number of control points on snake - MUST BE EVEN!


%%% Most relevant are ALPHA, BETA, GAMMA which weight the energy term for
%%% the snake (1st deriv, 2nd deriv, external field, respectively)

% Set this to ZERO if you have the image processing toolbox installed
% and are therefore able to use the bwdist function
USE_PRECOMPUTED_FIELD=1;

%%%%%%%%%%%% 1) Generate field for the snake
if (USE_PRECOMPUTED_FIELD) 
    field=double(imread('snakefield_synthetic.bmp'))./255;
else
    
    %%%%%%%%%%%% 1) Generate random test field for the snake
    t=0:(2*pi)/(JAGGEDNESS-1):2*pi;
    pts=100+[cos(t);sin(t)]*10;
    offset=0.5+rand(1,size(pts,2));
    pts=100+((pts-repmat(100,2,size(pts,2))).*repmat(offset,2,1)*BLOBSIZE);
    pts=[pts pts(:,1)];
    field=scancr(200,200,pts,1); 

    field=bwdist(field);

    field=double(1-pictNorm(field).^DECAY);  % the snake code assumes energy field in range [0,1]
end

%%%%%%%%%%% 2) Generate initial position of contour (in a big circle)
t=0:(2*pi)/(SNAKECP-1):2*pi;
INITCONTOUR=100+[cos(t);sin(t)]*90;

%%%%%%%%%%% 3) Perform snake relaxation
FITTEDCONTOUR=snkfit(field,INITCONTOUR,0);
plot([FITTEDCONTOUR(1,:) FITTEDCONTOUR(1,1)],[FITTEDCONTOUR(2,:) FITTEDCONTOUR(2,1)],'r');

%%%%%%%%%%% 4) visualise output
img(:,:,1)=field;
img(:,:,2)=field;
img(:,:,3)=field;

msk_fitted=scancr(200,200,[FITTEDCONTOUR FITTEDCONTOUR(:,1)],0); 
msk_init=scancr(200,200,[INITCONTOUR INITCONTOUR(:,1)],0); 

img(:,:,1)=max(img(:,:,1),msk_init);
img(:,:,2)=min(img(:,:,2),~msk_init);
img(:,:,3)=min(img(:,:,3),~msk_init);

img(:,:,1)=min(img(:,:,1),~msk_fitted);
img(:,:,2)=max(img(:,:,2),msk_fitted);
img(:,:,3)=min(img(:,:,3),~msk_fitted);
imgshow(img);
title('CR Spline Snakes (Williams/Shah)');


