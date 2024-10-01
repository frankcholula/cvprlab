close all;
clear all;

IMAGE_LEFT='images/sam_hp1.jpg';
IMAGE_RIGHT='images/sam_hp2.jpg';

img1=imread(IMAGE_LEFT);
img2=imread(IMAGE_RIGHT);

bothimg=[img1 img2];

imgshow(bothimg);
hold on;
title('Please click 8 pairs of points - left1, right1, left2, etc...');

Pleft=[];
Pright=[];
width=size(img1,2);
for pt=1:8

    [leftx lefty]=ginput(1);
    plot (leftx,lefty,'m+');

    [rightx righty]=ginput(1);
    plot (rightx,righty,'m+');
    
    line([leftx rightx],[lefty righty]);    
    
    Pleft=[Pleft [leftx;lefty]];
    Pright=[Pright [rightx-width;righty]];    
    
end
