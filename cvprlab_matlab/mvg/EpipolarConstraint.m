close all;

IMAGE_LEFT='images/sam_hp1.jpg';
IMAGE_RIGHT='images/sam_hp2.jpg';

img1=imread(IMAGE_LEFT);
img2=imread(IMAGE_RIGHT);

bothimg=[img1 img2];

imgshow(bothimg);
hold on;
title('Hit enter to continue...');

% Assume Pleft, Pright available (from Pick8Points)
% Assume F available (from EstimateF_8PointAlgorithm)
Pleft(3,:)=1;
Pright(3,:)=1;
width=size(img1,2);
for pt=1:8

    % Point on left is epipolar line on right
    left=Pleft(:,pt);
    plot(left(1),left(2),'r+');
    
    epipolar = left'*F;
    [st ed]=homogeneousline(epipolar);
    line([st(1)+width ed(1)+width],[st(2) ed(2)]);

    
    % Point on right is epipolar line on left
    right=Pright(:,pt);
    plot(right(1)+width,right(2),'g+');
    
    epipolar = F*right;
    [st ed]=homogeneousline(epipolar);
    line([st(1) ed(1)],[st(2) ed(2)]);    
    
    pause;
    
end
