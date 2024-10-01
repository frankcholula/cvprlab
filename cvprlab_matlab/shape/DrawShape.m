%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% DrawShape.m
%% Prompt the user to draw a 2D shape by clicking polygon vertices using
%% the mouse.  Return a binary mask of the filled polygon.
%%
%% Usage:  [mask, polyg] = DrawShape (size)
%%
%% IN:  size         - size of canvas e.g. 100 gives a 100x100 canvas
%%
%% OUT: mask         - binary mask of shape drawn
%%      polyg        - polygon vertices of shape drawn [x1 x2...;y1 y2...]
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom
 
function [mask pts]=DrawShape(canvassize)

% 1) Setup window ready to draw
figure;
image(uint8(ones(canvassize,canvassize,3).*255));
title('Click to draw a polygon (right click = last point)');
hold on;
axis on;
axis square;
axis ij;

% 2) Create empty matrix to store recorded points
% These will be of the form [x1 x2 x3.. xn;  y1 y2 y3.. yn]
pts=[];

% 2) Loop 
while (1)
    % get point
    [x y b]=ginput(1);
    
    % add the new point to pts
    pts=[pts [x;y]];

    % clear display
    cla;
    image(uint8(ones(canvassize,canvassize,3).*255));

    % plot points recorded so far as blue stars
    plot(pts(1,:),pts(2,:),'b*'); 
    
    % connect with blue lines
    for p=2:size(pts,2)        
        line([pts(1,p-1) pts(1,p)],[pts(2,p-1) pts(2,p)]); 
    end
    
    % highlight most recent point with a red circle
    plot(x,y,'ro');

    % if wasn't a left click then abort the loop
    if (b~=1)
        % complete the shape
        line([pts(1,end) pts(1,1)],[pts(2,end) pts(2,1)]); 
        drawnow;
        break;
    end       

end


mask=roipoly(zeros(canvassize,canvassize),pts(1,:)',pts(2,:)');