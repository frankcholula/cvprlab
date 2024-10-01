close all;
clear all;

IMAGE1='testimages/wall1.jpg';
IMAGE2='testimages/wall2.jpg';

img1=double(imread(IMAGE1))./255;
img2=double(imread(IMAGE2))./255;

gimg1=rgb2gray(img1);
gimg2=rgb2gray(img2);

% Load the sift results from Ex.1 - variables saved during Ex.1
load siftresults;

% Match the SIFT features between images
matches=siftmatch( descr1, descr2 ) ;

% Plot the keypoints
bothimages=[img1 img2];
image2offset=size(img1,2); % width of image 1
imgshow(bothimages);
hold on;

% M1 is the list of points matched from image1
% M2 is the list of points matched from image2
m1=matches(1,:);
m2=matches(2,:);

% Draw lines between matched points
STEPSIZE=50; % draw 1 in 10 lines
for match=1:STEPSIZE:length(m1)
   img1x=keypoints1(1,m1(match)); 
   img1y=keypoints1(2,m1(match)); 
   img2x=keypoints2(1,m2(match))+image2offset; 
   img2y=keypoints2(2,m2(match)); 
   plot(img1x,img1y,'yx');
   plot(img2x,img2y,'yx');
   line([img1x img2x],[img1y img2y]);
end
