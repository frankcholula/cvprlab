close all;
clear all;

IMAGE1='testimages/wall1.jpg';
IMAGE2='testimages/wall2.jpg';

img1=double(imread(IMAGE1))./255;
img2=double(imread(IMAGE2))./255;

gimg1=rgb2gray(img1);
gimg2=rgb2gray(img2);

% Run the SIFT detetor, and compute the SIFT descriptors
[frames1,descr1,gss1,dogss1] = sift( gimg1, 'Verbosity', 1 ) ;
[frames2,descr2,gss2,dogss2] = sift( gimg2, 'Verbosity', 1 ) ;
descr1=uint8(512*descr1) ;
descr2=uint8(512*descr2) ;
keypoints1=frames1(1:2,:);
keypoints2=frames2(1:2,:);

% Run the Harris corner detector
thresh=1000; % top 1000 corners
corners1 = torr_charris_jc(gimg1, thresh)';
corners2 = torr_charris_jc(gimg2, thresh)';


% Plot positions of the SIFT points and corners
figure(1);
imgshow(img1);
hold on;
plot(keypoints1(1,:),keypoints1(2,:),'yx');
plot(corners1(1,:),corners1(2,:),'bx');
title(IMAGE1);
% Plot positions of the SIFT points

% Plot positions of the SIFT points
figure(2);
imgshow(img2);
hold on;
plot(keypoints2(1,:),keypoints2(2,:),'yx');
plot(corners2(1,:),corners2(2,:),'bx');
title(IMAGE2);

% Helpful to give windows a chance to draw before disk tied up saving
% results
drawnow;

clear img1 img2 gimg1 gimg2;
save siftresults keypoints1 keypoints2 descr1 descr2;