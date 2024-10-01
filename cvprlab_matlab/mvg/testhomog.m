close all;
clear all;

IMG1='images/parade1.bmp';
IMG2='images/parade2.bmp';

imgshow(imread(IMG1));
title('LEFT:  Pick 4 points');
[x y]=ginput(4);
P=[x';y'];

imgshow(imread(IMG2));
title('RIGHT:  Pick 4 points');
[x y]=ginput(4);
Q=[x';y'];
