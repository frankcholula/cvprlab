close all;

IMAGE_LEFT='images/sam_hp1.jpg';
IMAGE_RIGHT='images/sam_hp2.jpg';

img1=imread(IMAGE_LEFT);
img2=imread(IMAGE_RIGHT);

bothimg=[img1 img2];

imgshow(bothimg);
hold on;
title('Click points to see epipolar lines...');

width=size(img1,2);
Pleft(3,:)=1;
Pright(3,:)=1;
while(1)

    [x y]=ginput(1);
    plot(x,y,'m+');
    
    if x>width
        % clicked on image 2
        right=[x-width;y;1];
        epipolar = F*right;
        [st ed]=homogeneousline(epipolar);
        line([st(1) ed(1)],[st(2) ed(2)]);    
        
    else
        % clicked on image 1
        left=[x;y;1];
        epipolar = left'*F;
        [st ed]=homogeneousline(epipolar);
        line([st(1)+width ed(1)+width],[st(2) ed(2)]);        
        
    end
      
end
